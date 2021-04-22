import time
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from . import logger as log
from . import resnet as models
from . import utils
from .debug import get_var, get_var_during_training
from actnn import config, QScheme, QModule, get_memory_usage, compute_tensor_bytes, exp_recorder
from copy import copy

try:
    # from apex.parallel import DistributedDataParallel as DDP
    from torch.nn.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


MB = 1024**2
GB = 1024**3

class ModelAndLoss(nn.Module):
    def __init__(self, arch, num_classes, loss, pretrained_weights=None, cuda=True, fp16=False):
        super(ModelAndLoss, self).__init__()
        self.arch = arch

        print("=> creating model '{}'".format(arch))
        model = models.build_resnet(arch[0], arch[1], num_classes)
        if arch[1] not in ['classic', 'fanin']:
            print("=> convert to quantized model")
            model = QModule(model)

        if pretrained_weights is not None:
            print("=> using pre-trained model from a file '{}'".format(arch))
            model.load_state_dict(pretrained_weights)

        if cuda:
            model = model.cuda()
        if fp16:
            model = network_to_half(model)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    def distributed(self, rank):
        self.model = DDP(self.model, device_ids=[rank])

    def load_model_state(self, state):
        if not state is None:
            try:
                self.model.load_state_dict(state)
            except:
                state = {k.replace('module.', ''): state[k] for k in state}
                self.model.load_state_dict(state)



def get_optimizer(parameters, fp16, lr, momentum, weight_decay,
                  nesterov=False,
                  state=None,
                  static_loss_scale=1., dynamic_loss_scale=False,
                  bn_weight_decay = False):

    if bn_weight_decay:
        print(" ! Weight decay applied to BN parameters ")
        optimizer = torch.optim.SGD([v for n, v in parameters], lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    nesterov = nesterov)
    else:
        print(" ! Weight decay NOT applied to BN parameters ")
        bn_params = [v for n, v in parameters if 'bn' in n]
        rest_params = [v for n, v in parameters if not 'bn' in n]
        print(len(bn_params))
        print(len(rest_params))
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay' : 0},
                                     {'params': rest_params, 'weight_decay' : weight_decay}],
                                    lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay,
                                    nesterov = nesterov)
    if fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=static_loss_scale,
                                   dynamic_loss_scale=dynamic_loss_scale,
                                   verbose=False)

    if not state is None:
        optimizer.load_state_dict(state)

    return optimizer


def lr_policy(lr_fn, logger=None):
    if logger is not None:
        logger.register_metric('lr', log.IterationMeter(), log_level=1)
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)

        if logger is not None:
            logger.log_metric('lr', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_linear_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = base_lr * (1-(e/es))
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_cosine_policy(base_lr, warmup_length, epochs, logger=None):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn, logger=logger)


def lr_exponential_policy(base_lr, warmup_length, epochs, final_multiplier=0.001, logger=None):
    es = epochs - warmup_length
    epoch_decay = np.power(2, np.log2(final_multiplier)/es)

    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            lr = base_lr * (epoch_decay ** e)
        return lr

    return lr_policy(_lr_fn, logger=logger)



def get_train_step(model_and_loss, optimizer, fp16, use_amp = False, batch_size_multiplier = 1):
    def _step(input, target, optimizer_step = True):
        input_var = Variable(input)
        target_var = Variable(target)

        if config.debug_memory_model:
            print("========== Init Data Loader ===========")
            init_mem = get_memory_usage(True)
            exp_recorder.record("data_loader", init_mem / GB - exp_recorder.val_dict['model_only'], 2)

        loss, output = model_and_loss(input_var, target_var)
        prec1, prec5 = torch.zeros(1), torch.zeros(1) #utils.accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
            #prec1 = reduce_tensor(prec1)
            #prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        if config.debug_memory_model:
            print("========== Before Backward ===========")
            before_backward = get_memory_usage(True)
            act_mem = get_memory_usage() - init_mem - compute_tensor_bytes([loss, output])
            res = "Batch size: %d\tTotal Mem: %.2f MB\tAct Mem: %.2f MB" % (
                    len(output), before_backward / MB, act_mem / MB)
            loss.backward()
            optimizer.step()
            del loss
            print("========== After Backward ===========")
            after_backward = get_memory_usage(True)
            total_mem = before_backward + (after_backward - init_mem)
            res = "Batch size: %d\tTotal Mem: %.2f MB\tAct Mem: %.2f MB" % (
                    len(output), total_mem / MB, act_mem / MB)
            print(res)
            exp_recorder.record("batch_size", len(output))
            exp_recorder.record("total", total_mem / GB, 2)
            exp_recorder.record("activation", act_mem / GB, 2)
            exp_recorder.dump('mem_results.tsv')
            exit()

        if fp16:
            optimizer.backward(loss)
        elif use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if optimizer_step:
            opt = optimizer.optimizer if isinstance(optimizer, FP16_Optimizer) else optimizer
            for param_group in opt.param_groups:
                for param in param_group['params']:
                    param.grad /= batch_size_multiplier

            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        return reduced_loss, output, prec1, prec5

    return _step

train_step_ct = 0
train_max_ips = 0
train_max_batch = 0

def train(train_loader, model_and_loss, optimizer, lr_scheduler, fp16, logger, epoch, use_amp=False, prof=-1, batch_size_multiplier=1, register_metrics=True):
    if register_metrics and logger is not None:
        logger.register_metric('train.top1', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.top5', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.loss', log.AverageMeter(), log_level = 0)
        logger.register_metric('train.compute_ips', log.AverageMeter(), log_level=1)
        logger.register_metric('train.total_ips', log.AverageMeter(), log_level=0)
        logger.register_metric('train.data_time', log.AverageMeter(), log_level=1)
        logger.register_metric('train.compute_time', log.AverageMeter(), log_level=1)

    if config.debug_memory_model:
        print("========== Model Only ===========")
        usage = get_memory_usage(True)
        exp_recorder.record("network", model_and_loss.arch[0])
        exp_recorder.record("algorithm", 'quantize'
                if model_and_loss.arch[1] == 'quantize' else 'exact')
        exp_recorder.record("model_only", usage / GB, 2)

    step = get_train_step(model_and_loss, optimizer, fp16, use_amp = use_amp, batch_size_multiplier = batch_size_multiplier)

    model_and_loss.train()
    print('Training mode ', config.training)
    end = time.time()

    optimizer.zero_grad()

    data_iter = enumerate(train_loader)
    if logger is not None:
        data_iter = logger.iteration_generator_wrapper(data_iter)

    for i, (input, target, index) in data_iter:     # NOTE: only needed for use_gradient
        QScheme.batch = index                       # NOTE: only needed for use_gradient

        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        if prof > 0:
            if i >= prof:
                break

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss, _, prec1, prec5 = step(input, target, optimizer_step = optimizer_step)

        it_time = time.time() - end

        if config.debug_speed:
            global train_step_ct, train_max_ips, train_max_batch
            train_max_ips = max(train_max_ips, calc_ips(bs, it_time))
            train_max_batch = max(train_max_batch, len(input))
            if train_step_ct >= 3:
                res = "BatchSize: %d\tIPS: %.2f\t,Cost: %.2f ms" % (
                    bs, train_max_ips, 1000.0 / train_max_ips)
                print(res, flush=True)
                exp_recorder.record("network", model_and_loss.arch[0])
                exp_recorder.record("algorithm", 'quantize'
                        if model_and_loss.arch[1] == 'quantize' else 'exact')
                exp_recorder.record("batch_size", train_max_batch)
                exp_recorder.record("ips", train_max_ips, 1)
                exp_recorder.dump('speed_results.tsv')
                exit(0)
            train_step_ct += 1

        if logger is not None:
            logger.log_metric('train.top1', to_python_float(prec1))
            logger.log_metric('train.top5', to_python_float(prec5))
            logger.log_metric('train.loss', to_python_float(loss))
            logger.log_metric('train.compute_ips', calc_ips(bs, it_time - data_time))
            logger.log_metric('train.total_ips', calc_ips(bs, it_time))
            logger.log_metric('train.data_time', data_time)
            logger.log_metric('train.compute_time', it_time - data_time)

        end = time.time()
        # if epoch > 0 and config.perlayer:
        #     QScheme.allocate_perlayer()
        #     QBNScheme.allocate_perlayer()

    #for layer in QScheme.layers:
    #    print(layer.name, layer.bits)


def get_val_step(model_and_loss):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():
            loss, output = model_and_loss(input_var, target_var)

        prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))

        if torch.distributed.is_initialized():
            reduced_loss = utils.reduce_tensor(loss.data)
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        return reduced_loss, prec1, prec5

    return _step


def validate(val_loader, model_and_loss, fp16, logger, epoch, prof=-1, register_metrics=True):
    if register_metrics and logger is not None:
        logger.register_metric('val.top1',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.top5',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.loss',         log.AverageMeter(), log_level = 0)
        logger.register_metric('val.compute_ips',  log.AverageMeter(), log_level = 1)
        logger.register_metric('val.total_ips',    log.AverageMeter(), log_level = 1)
        logger.register_metric('val.data_time',    log.AverageMeter(), log_level = 1)
        logger.register_metric('val.compute_time', log.AverageMeter(), log_level = 1)

    step = get_val_step(model_and_loss)

    top1 = log.AverageMeter()
    # switch to evaluate mode
    model_and_loss.eval()
    print('Training mode ', config.training)

    end = time.time()

    data_iter = enumerate(val_loader)
    if not logger is None:
        data_iter = logger.iteration_generator_wrapper(data_iter, val=True)

    for i, (input, target, _) in data_iter:
        bs = input.size(0)
        data_time = time.time() - end
        if prof > 0:
            if i > prof:
                break

        loss, prec1, prec5 = step(input, target)

        it_time = time.time() - end

        top1.record(to_python_float(prec1), bs)
        if logger is not None:
            logger.log_metric('val.top1', to_python_float(prec1))
            logger.log_metric('val.top5', to_python_float(prec5))
            logger.log_metric('val.loss', to_python_float(loss))
            logger.log_metric('val.compute_ips', calc_ips(bs, it_time - data_time))
            logger.log_metric('val.total_ips', calc_ips(bs, it_time))
            logger.log_metric('val.data_time', data_time)
            logger.log_metric('val.compute_time', it_time - data_time)

        end = time.time()

    return top1.get_val()

# Train loop {{{
def calc_ips(batch_size, time):
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    tbs = world_size * batch_size
    return tbs/time

def train_loop(model_and_loss, optimizer, new_optimizer, lr_scheduler, train_loader, val_loader, debug_loader, epochs, fp16, logger,
               should_backup_checkpoint, use_amp=False,
               batch_size_multiplier = 1,
               best_prec1 = 0, start_epoch = 0, prof = -1, skip_training = False, skip_validation = False, save_checkpoints = True, checkpoint_dir='./',
               model_state = None):
    QScheme.update_scale = True
    prec1 = -1

    epoch_iter = range(start_epoch, epochs)
    if logger is not None:
        epoch_iter = logger.epoch_generator_wrapper(epoch_iter)
    for epoch in epoch_iter:
        print('Epoch ', epoch)
        if not skip_training:
            train(train_loader, model_and_loss, optimizer, lr_scheduler, fp16, logger, epoch, use_amp = use_amp, prof = prof, register_metrics=epoch==start_epoch, batch_size_multiplier=batch_size_multiplier)

        if not skip_validation:
            prec1 = validate(val_loader, model_and_loss, fp16, logger, epoch, prof = prof, register_metrics=epoch==start_epoch)

        if save_checkpoints and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            if not skip_training:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)

                if should_backup_checkpoint(epoch):
                    backup_filename = 'checkpoint-{}.pth.tar'.format(epoch + 1)
                else:
                    backup_filename = None
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': model_and_loss.arch,
                    'state_dict': model_and_loss.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_dir=checkpoint_dir, backup_filename=backup_filename)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.end()

    get_var(model_and_loss, optimizer, train_loader, 10, model_state)
