import lasagne
from lasagne.layers.shape import PadLayer
from lasagne.layers.merge import ConcatLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, ReshapeLayer,\
    dropout, ElemwiseSumLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers.base import Layer
from lasagne import nonlinearities
from lasagne.layers.merge import MergeLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng

from lasagneCustomFingerprintLayers import LSTMLayer, SparsifyFingerprintLayer,\
    FingerprintHiddensLayer, FingerprintGenTopSum, VizAndPredMolLayer, FingerprintGen,\
    FingerprintMerge


import theano
import numpy as np
from theano.tensor import *
import theano.tensor as T


#################################################
#This is the functional layers for:
#    CNNvisualFingerprint.py
#################################################
def buildVisualCNNfingerprint(input_atom, input_bonds, input_atom_index, input_bond_index, input_mask, \
    max_atom_len, max_bond_len, input_atom_dim, input_bond_dim, input_index_dim, fingerprint_dim,
    batch_size, output_dim, final_layer_type, fingerprint_network_architecture=[],\
    final_hidden_units=0):

    dropout_prob = 0.0
    network_vals = {}

    #first encode one set of data
    l_in_atom = InputLayer(shape=(None,max_atom_len,input_atom_dim), input_var=input_atom)
    l_in_bond = InputLayer(shape=(None,max_bond_len,input_bond_dim), input_var=input_bonds)

    l_index_atom = InputLayer(shape=(None,max_atom_len,input_index_dim), input_var=input_atom_index)
    l_index_bond = InputLayer(shape=(None,max_atom_len,input_index_dim), input_var=input_bond_index)

    l_mask = InputLayer(shape=(None,max_atom_len), input_var=input_mask)


    first_hidden_units_num = fingerprint_network_architecture[0]

    #do the first hidden layer of the network
    network_vals['hiddens_for_atoms'] = FingerprintHiddensLayer([l_index_atom,l_index_bond,\
        l_in_atom,l_in_bond,l_in_atom,l_mask], input_atom_dim, input_atom_dim,\
        input_bond_dim, first_hidden_units_num, max_atom_len)

    #then sparsify
    network_vals['sparse_for_atoms'] = SparsifyFingerprintLayer([network_vals['hiddens_for_atoms']],\
        first_hidden_units_num, fingerprint_dim)

    #use this as the base of the fingerprints to sum to
    network_vals['fingerprints'] = network_vals['sparse_for_atoms']


    #then loop through all the other layers
    for i,curr_num_hiddens in enumerate(fingerprint_network_architecture[1:]):

        prev_hidden_units = fingerprint_network_architecture[i]

        #get the embedding of the sequences we are encoding
        network_vals['hiddens_for_atoms'] = FingerprintHiddensLayer([l_index_atom,\
            l_index_bond, l_in_atom, l_in_bond, network_vals['hiddens_for_atoms'],\
            l_mask], prev_hidden_units, input_atom_dim, input_bond_dim, \
            curr_num_hiddens, max_atom_len)

        new_fingerprints = SparsifyFingerprintLayer([network_vals['hiddens_for_atoms']],\
            first_hidden_units_num, fingerprint_dim)

        network_vals['fingerprints'] =  FingerprintGenTopSum([network_vals['fingerprints'],\
            new_fingerprints])


    #finally, project it into the dimensionality we want
    network_vals['output'] = VizAndPredMolLayer([network_vals['fingerprints'],\
        l_mask], num_input_units=fingerprint_dim, num_output_units=output_dim, \
        num_hidden_units=final_hidden_units, nonlinearity=final_layer_type)

    #return the dictionary of network values
    return network_vals



#################################################
#This is the functional layers for:
#    visualRNNfingerprints.py
#################################################
def rnnVisual(input_molecules, input_molecules_mask, num_features, max_len, \
    rnn_hidden_units, output_dim, final_layer_type,final_hidden_units=0):

    grad_clip = 100.0
    network_vals = {}

    #first encode one set of data
    l_in = InputLayer(shape=(None,max_len,num_features), input_var=input_molecules)
    l_mask = InputLayer(shape=(None,max_len), input_var=input_molecules_mask)

    lstm_forward = LSTMLayer(l_in, rnn_hidden_units,\
        mask_input=l_mask, learn_init=True, grad_clipping=grad_clip)
    lstm_backward = LSTMLayer(l_in, rnn_hidden_units,\
        mask_input=l_mask, learn_init=True, grad_clipping=grad_clip,\
        backwards=True)

    #get the embedding of the sequences we are encoding
    network_vals['embedding'] = ElemwiseSumLayer([lstm_forward, lstm_backward])

    network_vals['prediction'] = VizAndPredMolLayer([network_vals['embedding'],\
        l_mask],num_input_units=rnn_hidden_units, num_output_units=output_dim,
        num_hidden_units=final_hidden_units, nonlinearity=final_layer_type)

    return network_vals


#################################################
#This is the functional layers for:
#    deepRNNfingerprints.py
#################################################
def rnnControl(input_molecules, input_molecules_mask, num_features, max_len, \
    rnn_hidden_units, output_dim, final_layer_type):

    grad_clip = 100.0
    network_vals = {}

    #first encode one set of data
    l_in = InputLayer(shape=(None,max_len,num_features), input_var=input_molecules)
    l_mask = InputLayer(shape=(None,max_len), input_var=input_molecules_mask)

    rnn_forward = LSTMLayer(l_in, rnn_hidden_units, mask_input=l_mask, \
        learn_init=True, grad_clipping=grad_clip,)
    rnn_backward = LSTMLayer(l_in, rnn_hidden_units, mask_input=l_mask, \
        learn_init=True, grad_clipping=grad_clip, backwards=True)

    network_vals['first_layer'] = ElemwiseSumLayer([rnn_forward, rnn_backward])


    rnn_forward = LSTMLayer(network_vals['first_layer'], rnn_hidden_units,\
        mask_input=l_mask, learn_init=True, grad_clipping=grad_clip,)
    rnn_backward = LSTMLayer(network_vals['first_layer'], rnn_hidden_units,\
        mask_input=l_mask, learn_init=True, grad_clipping=grad_clip,\
        backwards=True)


    network_vals['second_layer'] = ElemwiseSumLayer([rnn_forward, rnn_backward])

    rnn_forward_ind = LSTMLayer(network_vals['second_layer'], rnn_hidden_units,\
        mask_input=l_mask, learn_init=True, grad_clipping=grad_clip,\
        only_return_final=True)
    rnn_backward_ind = LSTMLayer(network_vals['second_layer'], rnn_hidden_units,\
        mask_input=l_mask, learn_init=True, grad_clipping=grad_clip,\
        backwards=True, only_return_final=True)


    network_vals['third_layer'] = ElemwiseSumLayer([rnn_forward_ind, rnn_backward_ind])


    network_vals['prediction'] = DenseLayer(network_vals['third_layer'],
        output_dim, nonlinearity=final_layer_type)

    return network_vals


#################################################
#This is the functional layers for:
#    NNecfpControl.py
#################################################
def buildControlNN(fingerprint_vals, fingerprint_dim, output_dim, final_layer_type,
    dropout_prob=0.0, neural_net=[]):

    network_vals = {}

    l_in = InputLayer(shape=(None,fingerprint_dim), input_var=fingerprint_vals)

    #do dropout
    network_vals['drop0'] = dropout(l_in,p=dropout_prob)

    #run through the layers I have
    for layerNum,hiddenUnits in enumerate(neural_net):
        oldLayerNum = layerNum
        currLayerNum = layerNum + 1
        network_vals['dense'+str(currLayerNum)] = DenseLayer(network_vals['drop'+str(oldLayerNum)], \
            hiddenUnits,nonlinearity=lasagne.nonlinearities.rectify)
        network_vals['drop'+str(currLayerNum)] = dropout(network_vals['dense'+str(currLayerNum)],p=dropout_prob)


    if neural_net == []:
        network_vals['final_out'] = l_in
    else:
        network_vals['final_out'] = network_vals['dense'+str(currLayerNum)]


    #finally, project it into the dimensionality we want
    network_vals['output'] = DenseLayer(network_vals['final_out'],\
        num_units=output_dim, nonlinearity=final_layer_type)

    return network_vals


#################################################
#This is the functional layers for:
#    CNNfingerprint.py
#################################################
def buildCNNFingerprint(input_atom, input_bonds, input_atom_index, input_bond_index, input_mask, \
    max_atom_len, max_bond_len, input_atom_dim, input_bond_dim, input_index_dim, fingerprint_dim,
    batch_size, output_dim, final_layer_type, fingerprint_network_architecture=[],\
    neural_net=[]):

    dropout_prob = 0.0
    network_vals = {}

    #take in input layers for the atom and the bonds
    l_in_atom = InputLayer(shape=(None,max_atom_len,input_atom_dim), \
        input_var=input_atom)
    l_in_bond = InputLayer(shape=(None,max_bond_len,input_bond_dim), \
        input_var=input_bonds)

    #take in layers for the indexing into the atoms and bonds
    l_index_atom = InputLayer(shape=(None,max_atom_len,input_index_dim), \
        input_var=input_atom_index)
    l_index_bond = InputLayer(shape=(None,max_atom_len,input_index_dim), \
        input_var=input_bond_index)

    #take in input for the mask
    l_mask = InputLayer(shape=(None,max_atom_len), input_var=input_mask)


    #get the number of hidden units for the first layer
    first_hidden_units_num = fingerprint_network_architecture[0]


    #get the embedding of the sequences we are encoding
    network_vals['hiddens_for_atoms'] = FingerprintHiddensLayer([l_index_atom,l_index_bond,\
        l_in_atom,l_in_bond,l_in_atom,l_mask], input_atom_dim, input_atom_dim,
        input_bond_dim,first_hidden_units_num,max_atom_len)

    #sparsify
    network_vals['sparse_for_atoms'] = SparsifyFingerprintLayer([network_vals['hiddens_for_atoms']],\
        first_hidden_units_num, fingerprint_dim)


    network_vals['fingerprints'] = FingerprintGen([network_vals['sparse_for_atoms'],l_mask])

    for i,curr_num_hiddens in enumerate(fingerprint_network_architecture[1:]):

        prev_hidden_units = fingerprint_network_architecture[i]

        network_vals['hiddens_for_atoms'] = FingerprintHiddensLayer([l_index_atom,l_index_bond,\
            l_in_atom,l_in_bond,network_vals['hiddens_for_atoms'],l_mask], prev_hidden_units,
            input_atom_dim, input_bond_dim, curr_num_hiddens, max_atom_len)

        network_vals['sparse_for_atoms'] = SparsifyFingerprintLayer([network_vals['hiddens_for_atoms']],\
            curr_num_hiddens, fingerprint_dim)

        newFingerprints = FingerprintGen([network_vals['sparse_for_atoms'],l_mask])

        network_vals['fingerprints'] =  FingerprintMerge([network_vals['fingerprints'],newFingerprints])

    #then run through the neural net on top of the fingerprint
    if neural_net != []:
        #do dropout
        network_vals['drop0'] = lasagne.layers.dropout(network_vals['fingerprints'],p=dropout_prob)

        #run through the layers I have
        for layerNum,hiddenUnits in enumerate(neural_net):
            oldLayerNum = layerNum
            currLayerNum = layerNum + 1
            network_vals['dense'+str(currLayerNum)] = DenseLayer(network_vals['drop'+str(oldLayerNum)], \
                hiddenUnits,nonlinearity=lasagne.nonlinearities.rectify)
            network_vals['drop'+str(currLayerNum)] = dropout(network_vals['dense'+str(currLayerNum)],p=dropout_prob)

        network_vals['final_out'] = network_vals['drop'+str(currLayerNum)]

    else:
        network_vals['final_out'] = network_vals['fingerprints']


    #finally, project it into the dimensionality we want
    network_vals['output'] = DenseLayer(network_vals['final_out'],num_units=output_dim, nonlinearity=final_layer_type)

    return network_vals
