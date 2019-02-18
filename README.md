# Extracting musical features using Restricted Boltzmann Machines
An approach for unsupervised learning of musical features using Restricted Boltzmann Machines (RBMs) is investigated and measured against two other popular features for music. Specifically, we train four RBMs to represent a piece of music as a series of 10-, 30- or 50-dimensional vectors which can the be used for genre classification or other musical information retrieval tasks.

This is a project I did in 2011 during my master studies at the [Department of Computer Science, University of Copenhagen (DIKU)](http://diku.dk/). The code and the report is published here on Github for reading purposes only.

## Dependencies
The code depends on the [Shark Machine Learning Library](https://github.com/Shark-ML/Shark) for training RBMs and on [Marsyas](https://github.com/marsyas/marsyas) for the music information retrieval.

Besides the libraries, you also need the GTZAN genre collection dataset that consists og 1000 music audio files divided into 10 genres. The dataset can be retrieved here: http://marsyas.info/downloads/datasets.html

## Report
See [the report](https://github.com/andreas-hjortgaard/music_genre_classification/raw/master/musical_features_rbm.pdf) for a detailed description of the method and the results.
