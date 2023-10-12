# zetaPayne

This package is a simplified version of the Payne-Che package (https://github.com/istraumit/Payne-Che) with all its components structured into separate folders.


## Documentation

The paper introducing zeta-Payne: https://doi.org/10.3847/1538-3881/ac5f49


## Installation

Installation with Poetry: git clone the repository and run the command `poetry install` in the folder that contains the `pyproject.toml` and `poetry.lock` files.  


## Usage

The zeta-Payne package exists of 3 components:

1) Create a grid of model spectra.

2) Train a neural network on the grid.

3) Fit observed spectra using the neural network that functions as a model spectrum interpolator.


### step1_CreateGrid
Create a grid of model spectra with the spectrum synthesis code GSSP (https://ascl.net/2208.021).  
The grid can be distributed quasi-randomly (using Sobol numbers), randomly, or a combination of both.  
Update the configuration file `grid.conf` with the desired parameter boundary values, number of grid points, directory paths, ...

To create a quasi-random grid run: `python quasirandom_grid.py <path_to_config_file>`.  
To create a (additional) random grid run: `python random_grid.py <path_to_config_file>`.  
The output will be collected in a folder `grid` (or whatever is specified in the config file). These are the configuration file and all the model spectra as .npz files.

To assemble the whole grid into one file run: `python assemble_grid.py <path_to_grid>`.  
The assembled grid will be saved in a file called `GRID.npz` in the same grid folder.


### step2_TrainNN
Train a neural network on the grid of model spectra to create a spectrum interpolator that can predict a spectrum for any combination of surface parameters as long as they are within the boundaries of the grid.

The training of the neural networks requires GPUs.
To train the neural network run: `python train_NN.py <batch_size> <number_of_neurons> <path_to_GRID.npz> <lowest_wavelength> <highest_wavelength>`.  
The output neural network is saved in `NN_<number_of_neurons>_<batch_size>_<validation_fraction>.npz`.  

The neural network described in Gebruers et al. (2022) is available in step3_FitSpectra/NNs/NN_OPTIC_3000_10500_n300_b1000_v0.1.npz.  
This neural network was trained on a grid with the following parameter ranges: Teff in [6000,25000]K, logg in [3,5]dex, vsini in [0,400]km/s, microturbulence in [0,20]km/s, metallicity in [-0.8,+0.8]dex. The model spectra range from 3000 to 10500 angstrom in wavelength and have 'infinite' resolution.

### step3_FitSpectra
Fit observed spectra using a neural network to predict synthetic spectra. The spectrum normalisation is done simultaneously with the parameter determination by representing the response function with a series of Chebyshev polynomials of which the coefficients are added as fitting parameters.    
Update the configuration file `input.conf` with the desired wavelength range, number of Chebyshev coefficients, spectrograph resolution, ...

To fit an observed spectrum run: `python fitSpectrum.py <path_to_observed_spectrum>`.  
The output is saved in the `Output` folder. It will contain a folder with a file of the spectrum, the best fitting model spectrum, their normalised versions, and the Chebyshev polynomial series. There is also a LOG file saved with the best fitting surface parameters and a figure of the spectrum with best fit. 
