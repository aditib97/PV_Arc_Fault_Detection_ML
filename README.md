# Efficient Photovoltaic Arc Fault Detection with Machine Learning

### Author: Aditi Abhijit Bongale

The repository contains the code for the Master Thesis titled 'Efficient Photovoltaic Arc Fault Detection with Machine Learning'.

The required packages to be installed are in requirements.txt file.

### Data
Dataset for the project is provided at the following link: 
https://doi.org/10.5281/zenodo.14804124

Please create a folder ./data and unzip the archive into that directory.

### Usage
1) main.py runs the basic code right from data creation(labelling), data processing to extra trees classifer model validation.
2) classification_threshold.py performs a thresholding analysis for the dataset.
3) HPO.py is for hyperparameter tuning.
4) AN_main.py runs model evaluation using adaptive normalisation method.
5) AutoAN.py gives a comparison between automatic and manual adaptive normalisation.
6) WPE_main.py runs model evaluation using wavelet packet entropy features.
7) time_features_main.py runs model evaluation using common time-domain features.

Please create a folder ./results where the results of the code will be stored.