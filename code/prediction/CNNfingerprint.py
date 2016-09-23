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


expr_filename = '../../data/csv_files/logSolubilityTest.csv'
fingerprint_filename = '../../data/temp/logSolubilityInput_withRDKITidx.pkl'

#some hyperparameters of the job
#batch_size = 1000
batch_size = 100

#this is the dimension of the output fingerprint
fingerprint_dim = 265
#this is the dimension of the hiddens of the fingerprint
#the length of the list determines the number of layers for the molecule conv net
#fingerprint_network_architecture=[500]*5
fingerprint_network_architecture=[100]*2


#some hyperparameters
learning_rate = 0.001
num_epochs = 500
random_seed = int(time.time())
output_dim = 1
input_index_dim = 6
l2_regularization_lambda = 0.0001
final_layer_type = lasagne.nonlinearities.linear
start_time = str(time.ctime()).replace(':','-').replace(' ','_')

#this is a list that can arbitrarily make the neural neural on top of the CNN fingerprint
#it is a list that holds the dimensions of your neural network
neural_net=[1000]
#neural_net = [4000,3000,2000,1000]

#for no neural network on top, leave it empty
#neural_net=[]


#then make the name of the output to save for testing
neural_net_present = 'True'
if neural_net == []:
    neural_net_present = 'False'

#define the name of the output so I can save my predictions
if expr_filename == '../../data/csv_files/logSolubilityTest.csv':
    test_type = 'solubility'
progress_filename = 'log_files/CNN_fingerprint_NN-'+neural_net_present+'_'+test_type+'_'+start_time+'.csv'


#read in our drug data from seqHelper function also in this folder
smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
    smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask,\
    smiles_to_rdkit_list,max_atom_len,max_bond_len,num_atom_features,num_bond_features\
     = seqHelper.read_in_data(expr_filename,fingerprint_filename)

#grab the names of the experiments so I can make random test and train data
experiment_names = smiles_to_measurement.keys()

#get my random training and test set
test_num = int(float(len(experiment_names))*0.0)
train_num = int(float(len(experiment_names))*1.0)
test_list, train_list = seqHelper.gen_rand_train_test_data(experiment_names, \
    test_num, random_seed)

#define my theano variables
input_atom = ftensor3('input_atom')
input_atom_index = itensor3('input_atom_index')
input_bonds = ftensor3('input_bonds')
input_bond_index = itensor3('input_mask_attn')
input_mask = fmatrix('input_mask_attn')
target_vals = fvector('output_data')


#get my model output
cnn_model = lasagneModelsFingerprints.buildCNNFingerprint(input_atom, input_bonds, \
    input_atom_index, input_bond_index, input_mask, max_atom_len, max_bond_len, num_atom_features, \
    num_bond_features, input_index_dim, fingerprint_dim, batch_size, output_dim, final_layer_type, \
    fingerprint_network_architecture,neural_net)

print "Number of parameters:",lasagne.layers.count_params(cnn_model['output'])
print "batch size",batch_size
OUTPUT = open(progress_filename, 'w')
OUTPUT.write("NUM_PARAMS,"+str(lasagne.layers.count_params(cnn_model['output']))+'\n')
OUTPUT.write("EPOCH,RMSE,MSE\n")
OUTPUT.close()

#get the output of the model
train_prediction = lasagne.layers.get_output(cnn_model['output'],deterministic=False)
#flatten the prediction
train_prediction = train_prediction.flatten()
#define the loss as the trained error
train_loss = lasagne.objectives.squared_error(target_vals,train_prediction)

#regularize the network with L2 regularization
l2_loss = regularize_network_params(cnn_model['output'],l2)
train_cost = T.mean(train_loss) + l2_loss*l2_regularization_lambda

#get the parameters and updates from lasagne
params = lasagne.layers.get_all_params(cnn_model['output'], trainable=True)
updates = lasagne.updates.adam(train_cost, params, learning_rate=learning_rate)

#do this also for the test data
test_prediction = lasagne.layers.get_output(cnn_model['output'],deterministic=True)
test_prediction = test_prediction.flatten()
test_cost= lasagne.objectives.squared_error(target_vals,test_prediction)

#define our functions for train and test
train_func = theano.function([input_atom,input_bonds,input_atom_index,\
    input_bond_index,input_mask,target_vals], [train_cost], updates=updates, allow_input_downcast=True)

test_func = theano.function([input_atom,input_bonds,input_atom_index,\
    input_bond_index,input_mask,target_vals], [test_prediction,test_cost], allow_input_downcast=True)

print "compiled functions"

for epoch in xrange(num_epochs):

    #this function makes a list of lists that is the minibatch
    expr_list_of_lists = seqHelper.gen_batch_list_of_lists(train_list,batch_size,(random_seed+epoch))

    #then loop through the minibatches
    for counter,experiment_list in enumerate(expr_list_of_lists):
        x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val = seqHelper.gen_batch_XY_reg(experiment_list,\
            smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
            smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask)

        #run a training iteration
        train_error = train_func(x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val)
        print train_error
    test_error_list = []
    #every certain number of epochs, run the test data as well
    if epoch % 1 == 0:
        expr_list_of_lists = seqHelper.gen_batch_list_of_lists(test_list,batch_size,(random_seed+epoch))

        for experiment_list in expr_list_of_lists:
            x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val = seqHelper.gen_batch_XY_reg(experiment_list,\
                smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
                smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask)

            #run the test output
            test_prediction_output, test_error_output = test_func(x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val)

            print test_error_output
            #add it to the list of test error
            test_error_list += test_error_output.tolist()

        print "##########################################"
        print "EPOCH:\t"+str(epoch+1)+"\tRMSE\t",np.sqrt(np.mean(test_error_list)),'\tMSE\t',np.mean(test_error_list)
        print "##########################################"

        OUTPUT = open(progress_filename, 'a')
        OUTPUT.write(str(epoch)+","+str(np.sqrt(np.mean(test_error_list)))+','+str(np.mean(test_error_list))+'\n')
        OUTPUT.close()
