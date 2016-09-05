import lasagne
from lasagne.regularization import regularize_network_params, l2, l1

import time
import theano
import numpy as np
from theano.tensor import *
import theano.tensor as T


#then import my own modules
import seqHelper
import lasagneModelsFingerprints


#some hyperparameters
filename = '../../data/csv_files/logSolubilityTest.csv'
batch_size = 100
learning_rate = 0.001
num_epochs = 500
lstm_hidden_units = 500
random_seed = int(time.time())
output_dim = 1
l2_regularization_lambda = 0.00001
final_layer_type = lasagne.nonlinearities.linear
start_time = str(time.ctime()).replace(':','-').replace(' ','_')


#name the file that will save my progess
if filename == '../../data/csv_files/logSolubilityTest.csv':
    test_type = 'solubility'

progress_filename = 'output/RNN_deep-'+test_type+'_'+start_time+'.csv'


#read in my data
name_to_sequence,name_to_measurement,max_seq_length = seqHelper.read_in_chem_seq(filename)

#automatically make dictionaries that convert the alphabet of the letters to
#  one-hot encodings
alphabet_to_one_hot_sequence, num_in_one_hot_to_alphabet_sequence = \
    seqHelper.make_one_hot_dict(name_to_sequence)
#get the names of the sequences
sequence_names = name_to_sequence.keys()
#also get the number of letters in my alphabet
num_features = len(alphabet_to_one_hot_sequence)

#generate random test/train data
test_num = int(float(len(sequence_names))*0.2)
test_list, train_list = seqHelper.gen_rand_train_test_data(sequence_names, test_num, random_seed)

#save my theano variables
input_molecules = ftensor3('input_data')
input_molecules_mask = fmatrix('input_mask')
target_vals = fvector('output_data')

#get my random training and test set
test_num = int(float(len(sequence_names))*0.2)
trainNum = int(float(len(sequence_names))*0.8)
test_list, train_list = seqHelper.gen_rand_train_test_data(sequence_names, \
    test_num, random_seed)

#get my model
rnn_model = lasagneModelsFingerprints.rnnControl(input_molecules, \
    input_molecules_mask, num_features, max_seq_length, lstm_hidden_units,\
    output_dim, final_layer_type)


print "Number of parameters:",lasagne.layers.count_params(rnn_model['prediction'])
OUTPUT = open(progress_filename, 'w')
OUTPUT.write("NUM_PARAMS,"+str(lasagne.layers.count_params(rnn_model['prediction']))+'\n')
OUTPUT.write("EPOCH,RMSE,MSE\n")
OUTPUT.close()



#get our cost and our prediction
train_prediction = lasagne.layers.get_output(rnn_model['prediction'],deterministic=False)
train_prediction = train_prediction.flatten()

#make a squared error loss
train_loss = lasagne.objectives.squared_error(target_vals,train_prediction)

#regularize our net with L2 loss
l2_loss = regularize_network_params(rnn_model['prediction'],l2)
train_loss = lasagne.objectives.squared_error(train_prediction,target_vals)
train_cost = T.mean(train_loss) + l2_loss*l2_regularization_lambda

#get our parameters and our update from lasagne
params = lasagne.layers.get_all_params(rnn_model['prediction'], trainable=True)
updates = lasagne.updates.adam(train_cost, params, learning_rate=learning_rate)

#also get our test prediction as well
test_predicition = lasagne.layers.get_output(rnn_model['prediction'],deterministic=True)
test_predicition = test_predicition.flatten()
test_cost = lasagne.objectives.squared_error(target_vals,test_predicition)


train_func = theano.function([input_molecules,input_molecules_mask,\
    target_vals], [train_prediction,train_cost], updates=updates, allow_input_downcast=True)
test_func = theano.function([input_molecules,input_molecules_mask,\
    target_vals], [test_predicition,test_cost], allow_input_downcast=True)

print "compiled functions"


for epoch in xrange(num_epochs):
    #generate minibatches
    expr_list_of_lists_train = seqHelper.gen_batch_list_of_lists(train_list,batch_size,(random_seed+epoch))

    #then run through our minibatches
    for experiment_list in expr_list_of_lists_train:
        _,x_vals,x_mask,y_vals, = seqHelper.gen_batch_XY_rnn(experiment_list,name_to_sequence,\
            name_to_measurement,max_seq_length,alphabet_to_one_hot_sequence,output_dim)

        train_prediction,train_error = train_func(x_vals,x_mask,y_vals)

    #then run through the training iterations
    test_error_list = []
    if epoch % 1 == 0:
        expr_list_of_lists_test = seqHelper.gen_batch_list_of_lists(test_list,batch_size,(random_seed+epoch))

        for experiment_list in expr_list_of_lists_test:
            ascii_seq_list,x_vals,x_mask,y_vals = seqHelper.gen_batch_XY_rnn(experiment_list,name_to_sequence,\
                name_to_measurement,max_seq_length,alphabet_to_one_hot_sequence,output_dim)

            test_prediction,test_error = test_func(x_vals,x_mask,y_vals)
            test_error_list += test_error.tolist()

        #then write out my progress
        OUTPUT = open(progress_filename, 'a')
        OUTPUT.write(str(epoch)+","+str(np.sqrt(np.mean(test_error_list)))+','+str(np.mean(test_error_list))+'\n')
        OUTPUT.close()
