Visual Neural Graph Fingerprints in Theano
=============

<img src="https://github.com/debbiemarkslab/neural-fingerprint-theano/blob/master/data/images/neural_fing_img_github.png" width="700">

This software package implements visual convolutional neural graph fingerprints in Theano/Lasagne.

The papers that describe the improvements can be found at:

Visual Neural Graph Fingerprints in Theano

The original publication is located at:

[Convolutional Networks on Graphs for Learning Molecular Fingerprints](http://arxiv.org/pdf/1509.09292.pdf)

## Requirements

Prerequisites include:
* Numpy, Scipy, Matplotlib: These are all found in [Anaconda](https://www.continuum.io/downloads)
* [RDkit](http://www.rdkit.org/docs/Install.html)
* [Theano] (http://deeplearning.net/software/theano/install.html)
* [Lasagne](http://lasagne.readthedocs.io/en/latest/user/installation.html)

Please see inside the [code page](code/README.md) for specifics of usage.

## Examples

This code includes [regression examples](code/prediction) for visual convolutional neural fingerprints, convolutional neural fingerprints, LSTM-RNN on SMILES, and neural nets on standard fingerprints as well as the code to generate the [visualizations](code/visualization).

## Authors

The Theano/Lasagne implementation was written by [Adam Riesselman](ariesselman@g.harvard.edu). Please email me if you have any questions or concerns with this version.

The original neural graph fingerprint code was written by [David Duvenaud](http://www.cs.toronto.edu/~duvenaud/), [Dougal Maclaurin](mailto:maclaurin@physics.harvard.edu), and [Ryan P. Adams](http://www.seas.harvard.edu/directory/rpa); it can be found at their [Github](https://github.com/HIPS/neural-fingerprint).
