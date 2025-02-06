# mnist-in-core-mlx

This repo contains some basic code for classification with the MNIST dataset using core MLX. Note, that the file mnist.py is borrowed entirely from https://github.com/ml-explore/mlx-examples/blob/main/mnist/mnist.py. Run the above code with ```python run_example.py```.


In the above code, we compare two different loss -- the standard cross entropy loss and the harmonic loss described in https://arxiv.org/abs/2502.01628. For the particular choices of hyperparameters and using just basic SGD, we do not observe any significant difference between the training/test accuracies between the different loss functions. In fact, they track each other quite strongly. 

![accuracy](https://github.com/user-attachments/assets/ab8952c8-ae89-495b-90e3-3f5b7e5cc916)
