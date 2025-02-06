"""
This module contains training scripts for fitting simple MLP to MNIST, written to use only core MLX.
This example is akin to https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html,
with some modifications for fun.
"""
import time
from typing import Callable

import mlx.core as mx
import numpy as np

from mnist import mnist


def fit_model(seed: int, params: list[mx.array], batched_predict: Callable)->list:
    mx.random.seed(seed)
    # load data
    train_x, train_y, test_x, test_y = map(mx.array, mnist())

    # loss function and metrics
    def accuracy(
            params: list[mx.array],
            images: mx.array,
            targets: mx.array,
    ) -> mx.array:
        predicted_class = mx.argmax(batched_predict(params, images), axis=-1)
        return mx.mean(predicted_class == targets)

    def loss(
            params: list[mx.array],
            images: mx.array,
            targets: mx.array,
    ) -> mx.array:
        preds = batched_predict(params, images)
        # recall, preds contains log(probabilities) along axis=1
        # whereas targets contains the target class index
        return -mx.mean(preds[mx.arange(targets.size), targets])

    # model training settings
    learning_rate = 0.01
    num_epochs = 50
    batch_size = 128

    loss_and_grad_fn = mx.value_and_grad(loss, argnums=0)

    def batch_iterate(
            batch_size: int,
            X: mx.array,
            y: mx.array,
    ):
        perm = mx.array(np.random.permutation(y.size))
        for s in range(0, y.size, batch_size):
            ids = perm[s : s + batch_size]
            yield X[ids], y[ids]

    train_acc = []
    test_acc = []

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss = []
        for xb, yb in batch_iterate(batch_size, train_x, train_y):
            batch_loss, batch_grad = loss_and_grad_fn(params, xb, yb)

            # parameter update via SGD
            params = [
                (
                p[0] - learning_rate * g[0],
                p[1] - learning_rate * g[1],
                ) for p, g in zip(params, batch_grad)
            ]
            epoch_loss.append(batch_loss.item())

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_time = time.time() - start_time
        epoch_train_acc = accuracy(params, train_x, train_y)
        epoch_test_acc = accuracy(params, test_x, test_y)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Average training loss {}".format(avg_loss))
        print("Training set accuracy {}".format(epoch_train_acc))
        print("Test set accuracy {}".format(epoch_test_acc))

        train_acc.append(epoch_train_acc)
        test_acc.append(epoch_test_acc)

    return train_acc, test_acc

