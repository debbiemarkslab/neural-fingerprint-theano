Visual Neural Graph Fingerprints in Theano
=============

Neural Graph Fingerprints in Theano was designed for easy implementation in high-performance computing environments. The two main packages required to get this code working--Rdkit and Theano/Lasagne--can have different system, package, and Python requirements; a main design goal was to keep these two separate.

Rdkit is needed only briefly to generate molecule representations, while Theano and Lasagne do the heavy lifting in regards to deep learning.

## Package Details

Inside the code folder are three directories:

* [Rdkit Preprocessing](./rdkit_preprocessing) - Code that generates ECFP fingerprints as well as a pickled fingerprint data structure are generated here.
* [Prediction](./prediction) - This provides the guts of the learning algorithm. After Rdkit preprocessing, this code can be used to optimize any objective function relating to the activity of the molecule.
* [Visualization](./visualization) - Output from the [CNN Visual Fingerprints](./prediction/CNNvisualFingerprint.py) and [RNN Visual Fingerprint](./prediction/visualRNNfingerprints.py) can be run using this code.

## Instructions and Examples

Please see the [instructions page](./INSTRUCTIONS.txt) for examples on how to run the code.
