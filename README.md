# Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks


Sinusoidal neural networks have been shown effective as implicit neural
representations (INRs) of low-dimensional signals, due to their smoothness
and high representation capacity. However, initializing and training them
remain empirical tasks which lack on deeper understanding to guide the learning
process. To fill this gap, our work introduces a theoretical framework that
explains the capacity property of sinusoidal networks and offers robust control
mechanisms for initialization and training. Our analysis is based on a novel
amplitude-phase expansion of the sinusoidal multilayer perceptron, showing how
its layer compositions produce a large number of new frequencies expressed as
integer combinations of the input frequencies. This relationship can be
directly used to initialize the input neurons, as a form of spectral sampling,
and to bound the network’s spectrum while training. Our method, referred to as
TUNER (TUNing sinusoidal nEtwoRks), greatly improves the stability and
convergence of sinusoidal INR training, leading to detailed reconstructions, 
while preventing overfitting.

## Paper
Those interested in studying the theory and a more in-depth explanations of the
methods in this repository can check the paper on !!!


## Getting started
The implementation below considers that the user is familiarized with either SIREN [!!] implementation (which we'll denote as S) or with MR-Net [!!] repository (denoted as M).

### Code organization
Most of the code is organized in the `taming` package. Inside the corresponding folder, there are the several relevant folders such as:

* `configs`: Contains all the config files that are used to train the main scripts
* `data`: Stores the signal files such as images, meshes, etc.
* `runs`: Stores the results of the local experiments. For those using the wandb = True option on the config files, the 
data is stored on the corresponding Weights and Bias account and no local log is stored.
* `training`: Contains several scripts that are used on the training loop.

The files consider different initialization or training conditions:

* `train_siren.py`(S): Trains with the integer initialization described in !!!.
* `train_mrnet.py`(M): Trains with the integer initialization described in !!!.
* `train_bound.py`(M): Trains a model with integer initialization and fixed bounding scheme.
* `train_mrnet_learn_bounds.py`(M): Trains a model with integer initialization, learning the bounds.
* `train_tanh.py`(S): Trains a model with integer initialization, learning the bounds.
* `utils.py`: miscelaneous functions and utilities.

### Setup and sample run

1. Open a terminal (or Git Bash if using Windows)
2. Clone the repository: `git clone https://github.com/DianaPat/taming.git`.
3. Enter project folder: `cd taming`.
4. Create the environment and setup project dependencies:
```
pyenv virtualenv 3.9.9 i3d
pyenv local i3d
pip install -r requirements.txt
pip install -e .
```
5. Download the datasets (available [here](https://r0k.us/graphics/kodak/)) and extract them into the `data` folder of the repository
6. Train a network for the two macaws mesh:
```
python train_siren.py
```
7. The results will be stored in `runs/logs`, stored in a folder with a date ordering prefix and in another folder with the hour as prefix.

### End Result

The results of an experiment will return a prediction, the fourier transform of the approximation and the psnr.

<!-- ![Armadillo](figs/armadillo.png "Armadillo") -->

<!-- ### Linux

We tested the build steps stated above on Ubuntu 20.04. The prerequisites and setup remain the same, since all packages are available for both systems. We also provide a ```Makefile``` to cover the data download and network training and visualization (steps 5 through 9) above. -->



## Contact
If you have any questions, please feel free to email the authors, or open an issue.
