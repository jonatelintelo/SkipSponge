# AutoEncoders in PyTorch

## Description

This repo contains an implementation of the following AutoEncoders:

* [Vanilla AutoEncoders - **AE**](http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/): </br>
  The most basic autoencoder structure is one which simply maps input data-points through a __bottleneck layer__ whose dimensionality is smaller than the input.

* [Variational AutoEncoders - **VAE**](https://arxiv.org/pdf/1606.05908): </br>
  The Variational Autoencoder introduces the constraint
that the latent code `z` is a random variable distributed according to a prior distribution `p(z)`.

## Training
```
python train.py --help
```

### Training Options and some examples:

* **Vanilla Autoencoder:**
  ```
  python train.py --model AE
  ```

* **Variational Autoencoder:**
  ```
  python train.py --model VAE --batch-size 512 --dataset EMNIST --seed 42 --log-interval 500 --epochs 5 --embedding-size 128
  ```
Please check parser arguments for all possible parameters and default values.
