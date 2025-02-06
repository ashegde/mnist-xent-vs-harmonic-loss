"""
In this script, we compare two MLPs -- one using a cross-entropy loss, and another using a harmonic loss --
on MNIST. The harmonic loss is based on https://arxiv.org/abs/2502.01628:
"""
import copy 

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from model import build_params, predict_single_xent, predict_single_harmonic
from fitting import fit_model

seed = 2025

layer_sizes = [784, 512, 512, 10]
params_xent = build_params(seed, layer_sizes=layer_sizes)
batched_predict_xent = mx.vmap(predict_single_xent, in_axes=(None, 0))
params_harmonic = build_params(seed, layer_sizes=layer_sizes)
batched_predict_harmonic = mx.vmap(predict_single_xent, in_axes=(None, 0))

print('\nCASE 1: CROSS-ENTROPY LOSS\n')
train_acc_xent, test_acc_xent = fit_model(seed, params_xent, batched_predict_xent)

print('\nCASE 2: HARMONIC LOSS\n')
train_acc_harmonic, test_acc_harmonic = fit_model(seed, params_harmonic, batched_predict_harmonic)

# plots
epochs = range(len(train_acc_harmonic))
plt.plot(epochs, train_acc_harmonic, c='r', linestyle='--', label='train-harmonic')
plt.plot(epochs, test_acc_harmonic, c='r', label='test-harmonic')
plt.plot(epochs, train_acc_xent, c='b', linestyle='--', label='train-xent')
plt.plot(epochs, test_acc_xent, c='b', label='test-xent')
plt.legend(loc='lower right')
plt.title('Accuracy during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yscale('log')
plt.savefig('accuracy.png')


