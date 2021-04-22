import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import seaborn as sns

weight_names = pickle.load(open('layer_names.pkl', 'rb'))

grads = np.load('error_profile_5.npy')
grads = np.maximum(grads, 0)
quantizers = grads.sum(1)
variances = grads.sum(0)

# grads = np.minimum(grads, 1)
# grads *= 1000
for i in range(grads.shape[0]):
    for j in range(grads.shape[1]):
        if j > i:
            grads[i, j] = 0

fig, ax = plt.subplots(figsize=(20, 20))
im = ax.imshow(grads, cmap='Blues', norm=LogNorm(vmin=0.01, vmax=10.0))
ax.set_xticks(np.arange(len(weight_names)))
ax.set_yticks(np.arange(len(weight_names)+1))
ax.set_xticklabels(weight_names)
weight_names.append('sample')
ax.set_yticklabels(weight_names)

ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)
plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
         rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax)

for i in range(grads.shape[0]):
    for j in range(grads.shape[1]):
        text = ax.text(j, i, int(grads[i, j]*10),
                       ha="center", va="center")


fig.savefig('variance_profile.pdf')

fig, ax = plt.subplots(figsize=(20, 20))
sns.barplot(x=np.arange(quantizers.shape[0]), y=quantizers, ax=ax)
ax.set_xticks(np.arange(len(weight_names)) + 0.5)
ax.set_xticklabels(weight_names)
plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
         rotation_mode="anchor")
ax.set_xlabel('quantizer')
ax.set_ylabel('variance')
fig.savefig('quantizers.pdf')

fig, ax = plt.subplots(figsize=(20, 20))
sns.barplot(x=np.arange(variances.shape[0]), y=variances, ax=ax)
weight_names.pop(-1)
ax.set_xticks(np.arange(len(weight_names)) + 0.5)
ax.set_xticklabels(weight_names)
plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
         rotation_mode="anchor")
ax.set_xlabel('parameter')
ax.set_ylabel('variance')
fig.savefig('parameter.pdf')
