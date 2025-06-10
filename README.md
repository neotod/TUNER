# Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks

This is the official repository of the accepted CVPR paper. We use the discovered 
theoretical framework to initialize the input neurons, as a form of spectral sampling,
and to bound the network’s spectrum while training. Our method, referred to as
TUNER (TUNing sinusoidal nEtwoRks), greatly improves the stability and
convergence of sinusoidal INR training, leading to detailed reconstructions, 
while preventing overfitting. For more details, access the links below

<div style="display: flex; gap: 20px;">
    <a href="https://DianaPat.github.io/tuner/"><img src="https://img.shields.io/badge/Button_Label-blue" alt="Page"></a>
  <a href="docs/assets/Novello_Tuning_the_Frequencies_Robust_Training_for_Sinusoidal_Neural_Networks_CVPR_2025_paper.pdf"><img src="https://img.shields.io/badge/Button_3-red" alt="Paper"></a>
</div>

## Getting started
The implementation below considers that the user is familiarized with either SIREN ![SIREN](https://www.vincentsitzmann.com/siren/) or MR-Net ![MR-Net](https://www.sciencedirect.com/science/article/pii/S0097849323000699) repository.


### Code organization
The code is organized as follows:

* `ablations`: Some of the scripts used to run experiments of the main paper
* `configs`: Contains the config file that sets the hyperparameters used to train the scripts
* `data`: Datasets used to test and train the models.
* `networks`: Scripts with the classes for the two architectures of SIREN and MRNet
* `training`: Scripts to train MRNet, methods to log the results

### Setup and sample run

1. Open a terminal (or Git Bash if using Windows)
2. Clone the repository: `git clone https://github.com/DianaPat/taming.git`.
3. Enter project folder: `cd tuner`.
4. Create the environment and setup project dependencies
5. Download the kodak datasets (available [here](https://r0k.us/graphics/kodak/)) and extract them into the `data` folder of the repository
6. Train a network for the two macaws mesh:
```
python train_siren.py
```
7. If the `logger` is set as `local`, the results will be logged in
`runs/logs`, stored in a folder with a date ordering prefix and in another
folder with the hour as prefix.

## Contact
If you have any questions, please feel free to email the authors, or open an issue.
