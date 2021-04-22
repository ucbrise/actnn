from torchvision.models.resnet import _resnet, Bottleneck

def scaled_resnet(name):
    n_layers = int(name.split('scaled_resnet_')[1])
    assert n_layers >= 152
    added_layers = n_layers - 152

    add_1 = int(added_layers * (8 / (8 + 36)))
    add_2 = added_layers - add_1

    return _resnet('', Bottleneck, [3, 8 + add_1, 36 + add_2, 3], False, True)

def scaled_wide_resnet(name):
    width = int(name.split('scaled_wide_resnet_')[1])
    kwargs = {'width_per_group': width}
    return _resnet('', Bottleneck, [3, 8, 36, 3], False, True, **kwargs)

