import lasagne
from lasagne.regularization import regularize_network_params, l2, l1
import time
import theano
import numpy as np
from theano.tensor import *
import theano.tensor as T
import gc
import sys
sys.setrecursionlimit(50000)

#then import my own modules
import seqHelper
import lasagneModelsFingerprints


#get my files
filename = '../../data/csv_files/logSolubilityTest.csv'
fingerprint_filename = '../../data/temp/logSolubilityInput_withRDKITidx.pkl'

#set some hyperparameters
batch_size = 100
learning_rate = 0.001
num_epochs = 500
random_seed = int(time.time())
output_dim = 1
input_index_dim = 6
l2_regularization_lambda = 0.0001
final_layer_type = lasagne.nonlinearities.linear
start_time = str(time.ctime()).replace(':','-').replace(' ','_')

#this is the dimension of the output fingerprint
fingerprint_dim = 265
#this is the dimension of the hiddens of the fingerprint
#the length of the list determines the number of layers for the molecule conv net
fingerprint_network_architecture=[500]*5

# for a neural net on the final output, make this number > 0
final_neural_net = 1000

#otherwise, set it to 0 if just a linear layer is desired
#final_neural_net = 0


#then make the name of the output to save for testing
neural_net_present = 'True'
if neural_net == []:
    neural_net_present = 'False'

if expr_filename == 'data/logSolubilityTest.csv':
    test_type = 'solubility'

progress_filename = 'output/CNN_fingerprint_visual_context_NN-'+neural_net_present+'_'+test_type+'_'+start_time+'.csv'



#read in our molecule data
smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
    smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask,\
    smiles_to_rdkit_list,max_atom_len,max_bond_len,num_atom_features,num_bond_features\
     = seqHelper.read_in_data(expr_filename,fingerprint_filename)

#then get some variables ready to set up my model
experiment_names = smiles_to_measurement.keys()

#get my random training and test set
test_num = int(float(len(experiment_names))*0.2)
train_num = int(float(len(experiment_names))*0.8)
test_list, train_list = seqHelper.gen_rand_train_test_data(experiment_names, test_num, random_seed)

#define my theano variables
input_atom = ftensor3('input_atom')
input_atom_index = itensor3('input_atom_index')
input_bonds = ftensor3('input_bonds')
input_bond_index = itensor3('input_mask_attn')
input_mask = fmatrix('input_mask_attn')
target_vals = fvector('output_data')



#get my model output
cnn_model = lasagneModelsFingerprints.buildVisualCNNfingerprint(input_atom, input_bonds, \
    input_atom_index, input_bond_index, input_mask, max_atom_len, max_bond_len, num_atom_features, \
    num_bond_features, input_index_dim, fingerprint_dim, batch_size, output_dim, final_layer_type, \
    fingerprint_network_architecture,final_neural_net)

#count the number of parameters in the model and initialize progress file
print "Number of parameters:",lasagne.layers.count_params(cnn_model['output'])
OUTPUT = open(progress_filename, 'w')
OUTPUT.write("NUM_PARAMS,"+str(lasagne.layers.count_params(cnn_model['output']))+'\n')
OUTPUT.write("EPOCH,RMSE,MSE\n")
OUTPUT.close()

#mulitply our training predictions and visualizations by 1.0
#   this makes it so these numbers are part of the theano graph but not changed in value
#   in this way, theano doesn't complain at me for unused variables
context_output_train = lasagne.layers.get_output(cnn_model['output'],deterministic=False)
train_prediction = context_output_train[0] * 1.0
visual_predictions_train = context_output_train[1] * 1.0
train_prediction = train_prediction.flatten()
train_loss = lasagne.objectives.squared_error(target_vals,train_prediction)

#get our loss and our cost
l2_loss = regularize_network_params(cnn_model['output'],l2)
train_cost = T.mean(train_loss) + l2_loss*l2_regularization_lambda

#then get our parameters and update from lasagne
params = lasagne.layers.get_all_params(cnn_model['output'], trainable=True)
updates = lasagne.updates.adam(train_cost, params, learning_rate=learning_rate)

#then get the outputs for the test and multiply them by 1.0 like above
context_output_test = lasagne.layers.get_output(cnn_model['output'],deterministic=True)
test_predicition = context_output_test[0] * 1.0
visual_predictions_test = context_output_test[1] * 1.0
test_predicition = test_predicition.flatten()
test_cost = lasagne.objectives.squared_error(target_vals,test_predicition)

#then define my theano functions for train and test
train_func = theano.function([input_atom,input_bonds,input_atom_index,\
    input_bond_index,input_mask,target_vals], [train_prediction,train_cost,visual_predictions_train],\
    updates=updates, allow_input_downcast=True)

test_func = theano.function([input_atom,input_bonds,input_atom_index,\
    input_bond_index,input_mask,target_vals], [test_predicition,test_cost,visual_predictions_test], allow_input_downcast=True)

print "compiled functions"

#then run through my epochs
for epoch in xrange(num_epochs):

    #get my minibatch
    expr_list_of_lists_train = seqHelper.gen_batch_list_of_lists(train_list,batch_size,(random_seed+epoch))

    #run through my training minibatches
    for counter,experiment_list in enumerate(expr_list_of_lists_train):
        x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val = seqHelper.gen_batch_XY_reg(experiment_list,\
            smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
            smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask)

        train_prediction,train_error,train_viz = train_func(x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val)


    test_error_list = []
    if epoch % 1 == 0:
        expr_list_of_lists_test = seqHelper.gen_batch_list_of_lists(test_list,batch_size,(random_seed+epoch))

        #then run through the test data
        OUTPUTVIZ = open('output/'+test_type+'_chemotype_predictions_NN-'+neural_net_present+'_'+start_time+'.csv', 'w')
        for experiment_list in expr_list_of_lists_test:
            x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val = seqHelper.gen_batch_XY_reg(experiment_list,\
                smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
                smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask)

            #do the prediction
            test_prediction,test_error_output,test_viz = test_func(x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val)

            #add the error to the error list
            test_error_list += test_error_output.tolist()

            #write out my visual predictions
            seqHelper.write_out_predictions_cnn(experiment_list,x_mask,smiles_to_rdkit_list, test_viz, OUTPUTVIZ)

        #then also do the visualizations for the training data
        for experiment_list in expr_list_of_lists_train:
            x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val = seqHelper.gen_batch_XY_reg(experiment_list,\
                smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
                smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask)

            test_prediction,test_error_output,test_viz = test_func(x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val)

            seqHelper.write_out_predictions_cnn(experiment_list,x_mask, smiles_to_rdkit_list, test_viz, OUTPUTVIZ)

        OUTPUTVIZ.close()

        #then write out my progress
        OUTPUT = open(progress_filename, 'a')
        OUTPUT.write(str(epoch)+","+str(np.sqrt(np.mean(test_error_list)))+','+str(np.mean(test_error_list))+'\n')
        OUTPUT.close()
