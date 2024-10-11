# Adversarial Autoencoders

* `mnist.py`: preparing the dataset and providing a data loader for training.
* `models.py`: Encoder, Decoder, Discriminator and the overall Adversarial Autoencoder.
* `train.py`: Contains the overall training procedure sets the hyper-parameters, load dataset, initialize the model, set the optimizers and then it trains the adversarial auto-encoder and saves the network generations for each epoch.   
* `utils.py`: Contains logging utilities for Tensorboard.

![Reconstruction Adversarial](images/reconstruction_adversarial.png)
![Reconstruction Standard](images/reconstruction_standard.png)