import lasagne
from lasagne.layers.shape import PadLayer
from lasagne.layers.merge import ConcatLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, ReshapeLayer, dropout
from lasagne.nonlinearities import softmax, rectify
#from lasagne.layers.normalization import BatchNormLayer, batch_norm
from lasagne.layers.base import Layer
from lasagne.regularization import regularize_network_params, l2, l1

import time
import theano
import numpy as np
from theano.tensor import *
import theano.tensor as T
import gc


#then import my own modules
import seqHelper
import lasagneModelsFingerprints


#get the name of files we need, which is the csv files of the activity...
expr_filename = '../../data/csv_files/logSolubilityTest.csv'

#and the name of the fingerprints
fingerprint_filename = '../../data/temp/solubility_control_fingerprints_2048.csv'

#then get all the hyperparameters as well
batch_size = 100
learning_rate = 0.001
num_epochs = 300
fingerprint_dim = 2048
random_seed = int(time.time())
l2_regularization_lambda = 0.0001
final_layer_type = lasagne.nonlinearities.linear
output_dim = 1
dropout_prob = 0.0
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


if expr_filename == '../../data/csv_files/logSolubilityTest.csv':
    test_type = 'solubility'

progress_filename = 'output/NN_control-'+neural_net_present+'_'+test_type+'_'+start_time+'.csv'

#read in our drug data
smiles_to_prediction,smiles_to_fingerprint \
     = seqHelper.read_in_ecfp_data(expr_filename,fingerprint_filename)

#then get some variables ready to set up my model
experiment_names = smiles_to_prediction.keys()

#get my random training and test set
test_num = int(float(len(experiment_names))*0.2)
train_num = int(float(len(experiment_names))*0.8)
test_list, train_list = seqHelper.gen_rand_train_test_data(experiment_names, test_num, random_seed)

#define my theano variables
input_fingerprints = fmatrix('input_fingerprints')
target_vals = fvector('output_data')


#get my model output
nn_model = lasagneModelsFingerprints.buildControlNN(input_fingerprints,
    fingerprint_dim, output_dim, final_layer_type, dropout_prob, neural_net)

print "Number of parameters:",lasagne.layers.count_params(nn_model['output'])
OUTPUT = open(progress_filename, 'w')
OUTPUT.write("NUM_PARAMS,"+str(lasagne.layers.count_params(nn_model['output']))+'\n')
OUTPUT.write("EPOCH,RMSE,MSE\n")
OUTPUT.close()

#get my training prediction
train_prediction = lasagne.layers.get_output(nn_model['output'],deterministic=False)
train_prediction = train_prediction.flatten()
train_loss = lasagne.objectives.squared_error(target_vals,train_prediction)

#get my loss with l2 regularization
l2_loss = regularize_network_params(nn_model['output'],l2)
train_cost = T.mean(train_loss) + l2_loss*l2_regularization_lambda

#then my parameters and updates from theano
params = lasagne.layers.get_all_params(nn_model['output'], trainable=True)
updates = lasagne.updates.adam(train_cost, params, learning_rate=learning_rate)

#pull out my test predictions
test_prediction = lasagne.layers.get_output(nn_model['output'],deterministic=True)
test_prediction = test_prediction.flatten()
test_cost= lasagne.objectives.squared_error(target_vals,test_prediction)

#then get my training and test functions
train_func = theano.function([input_fingerprints,target_vals], \
    [train_prediction,train_cost], updates=updates, allow_input_downcast=True)

test_func = theano.function([input_fingerprints,target_vals], \
    [test_prediction,test_cost], allow_input_downcast=True)

print "compiled functions"

for epoch in xrange(num_epochs):

    #generate the training minibatch
    expr_list_of_lists = seqHelper.gen_batch_list_of_lists(train_list,batch_size,(random_seed+epoch))

    for counter,experiment_list in enumerate(expr_list_of_lists):

        #generate my minibatch parameters for X and Y
        x_fing, y_val = seqHelper.gen_batch_XY_control(experiment_list,\
            smiles_to_prediction,smiles_to_fingerprint)

        #run a training iteration
        train_output_pred,train_error = train_func(x_fing, y_val)


    #then run my test
    test_error_list = []
    if epoch % 1 == 0:

        #generate the test minibatch
        expr_list_of_lists = seqHelper.gen_batch_list_of_lists(test_list,batch_size,(random_seed+epoch))


        #then run through the minibatches
        for experiment_list in expr_list_of_lists:
            x_fing, y_val = seqHelper.gen_batch_XY_control(experiment_list,\
                smiles_to_prediction,smiles_to_fingerprint)

            #get out the test predictions
            test_output_pred,test_error_output = test_func(x_fing, y_val)

            test_error_list += test_error_output.tolist()

        #then write our output
        OUTPUT = open(progress_filename, 'a')
        OUTPUT.write(str(epoch)+","+str(np.sqrt(np.mean(test_error_list)))+','+str(np.mean(test_error_list))+'\n')
        OUTPUT.close()
        print (str(epoch)+","+str(np.sqrt(np.mean(test_error_list)))+','+str(np.mean(test_error_list)))
