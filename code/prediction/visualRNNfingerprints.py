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


#some hyperparameters
filename = '../../data/csv_files/logSolubilityTest.csv'

batch_size = 100
learning_rate = 0.001
num_epochs = 300
rnn_hidden_units = 500
random_seed = int(time.time())
output_dim = 1
l2_regularization_lambda = 0.00001
final_layer_type = lasagne.nonlinearities.linear
start_time = str(time.ctime()).replace(':','-').replace(' ','_')

# for a neural net on the final output, make this number > 0
final_neural_net = 1000

#otherwise, set it to 0 if just a linear layer is desired
#final_neural_net = 0

#then make the name of the output to save for testing
neural_net_present = 'True'
if neural_net == []:
    neural_net_present = 'False'

if filename == '../../data/csv_files/logSolubilityTest.csv':
    test_type = 'solubility'

#then open a progress file
progress_filename = 'output/RNN_visual_context_NN-'+neural_net_present+'_'+test_type+'_'+start_time+'.csv'


name_to_sequence,name_to_measurement,max_seq_length = seqHelper.read_in_chem_seq(filename)

alphabet_to_one_hot_sequence, num_in_one_hot_to_alphabet_sequence = \
    seqHelper.make_one_hot_dict(name_to_sequence)

#get the names of my sequences
sequence_names = name_to_sequence.keys()

#get the number of features which is my alphabet size
num_features = len(alphabet_to_one_hot_sequence)

#get a test and train set that is random
test_num = int(float(len(sequence_names))*0.2)
test_list, train_list = seqHelper.gen_rand_train_test_data(sequence_names, test_num, random_seed)

#initialize my theano variables
input_molecules = ftensor3('input_data')
input_molecules_mask = fmatrix('input_mask')
target_vals = fvector('output_data')

#get my random training and test set
test_num = int(float(len(sequence_names))*0.2)
trainNum = int(float(len(sequence_names))*0.8)
test_list, train_list = seqHelper.gen_rand_train_test_data(sequence_names, \
    test_num, random_seed)

#get our model
context_rnn_model = lasagneModelsFingerprints.rnnVisual(input_molecules, \
    input_molecules_mask, num_features, max_seq_length, rnn_hidden_units,\
    output_dim, final_layer_type, final_neural_net)


#count the number of parameters in the model and initialize progress file
print "Number of parameters:",lasagne.layers.count_params(context_rnn_model['prediction'])
OUTPUT = open(progress_filename, 'w')
OUTPUT.write("NUM_PARAMS,"+str(lasagne.layers.count_params(context_rnn_model['prediction']))+'\n')
OUTPUT.write("EPOCH,RMSE,MSE\n")
OUTPUT.close()

#pull our output from the layers
context_output_train = lasagne.layers.get_output(context_rnn_model['prediction'],deterministic=False)

#mulitply our training predictions and visualizations by 1.0
#   this makes it so these numbers are part of the theano graph but not changed in value
#   in this way, theano doesn't complain at me for unused variables
train_prediction = context_output_train[0] * 1.0
visual_predictions_train = context_output_train[1] * 1.0
train_prediction = train_prediction.flatten()
train_loss = lasagne.objectives.squared_error(target_vals,train_prediction)

l2_loss = regularize_network_params(context_rnn_model['prediction'],l2)

#get our loss and our cost
train_loss = lasagne.objectives.squared_error(train_prediction,target_vals)
train_cost = T.mean(train_loss) + l2_loss*l2_regularization_lambda

#then get our parameters and updates
params = lasagne.layers.get_all_params(context_rnn_model['prediction'], trainable=True)
updates = lasagne.updates.adam(train_cost, params, learning_rate=learning_rate)

#then get the outputs for the test and multiply them by 1.0 like above
context_output_test = lasagne.layers.get_output(context_rnn_model['prediction'],deterministic=True)
test_predicition = context_output_test[0] * 1.0
visual_predictions_test = context_output_test[1] * 1.0
test_predicition = test_predicition.flatten()
test_cost = lasagne.objectives.squared_error(target_vals,test_predicition)

#get my theano funtions
train_func = theano.function([input_molecules,input_molecules_mask,\
    target_vals], [train_prediction,train_cost,visual_predictions_train], updates=updates, allow_input_downcast=True)
test_func = theano.function([input_molecules,input_molecules_mask,\
    target_vals],[test_predicition,test_cost,visual_predictions_test], allow_input_downcast=True)

print "compiled functions"

#now run through all my epochs
for epoch in xrange(num_epochs):

    #generate my training minibatches
    expr_list_of_lists_train = seqHelper.gen_batch_list_of_lists(train_list,batch_size,(epoch+random_seed))

    for experiment_list in expr_list_of_lists_train:
        _,x_vals,x_mask,y_vals, = seqHelper.genBatchXYchem(experiment_list,name_to_sequence,\
            name_to_measurement,max_seq_length,alphabet_to_one_hot_sequence,output_dim)

        #then do the training
        train_prediction,train_error,train_viz = train_func(x_vals,x_mask,y_vals)


    testErrorList = []
    if epoch % 1 == 0:

        #generate my minibatches
        expr_list_of_lists_test = seqHelper.gen_batch_list_of_lists(test_list,batch_size,(epoch+random_seed))
        OUTPUT = open('output/'+test_type+'_chemotype_predictions_RNN_NN-'+neural_net_present+'_'+start_time+'.csv', 'w')

        for experiment_list in expr_list_of_lists_test:
            ascii_seq_list,x_vals,x_mask,y_vals = seqHelper.gen_batch_XY_rnn(experiment_list,name_to_sequence,\
                name_to_measurement,max_seq_length,alphabet_to_one_hot_sequence,output_dim)

            #run the test
            test_prediction,test_error_output,test_viz = test_func(x_vals,x_mask,y_vals)

            #get my error and save it to a list
            testErrorList += test_error_output.tolist()

            #then write out the prediction for the visualization for test
            seqHelper.write_out_rnn_prediction(experiment_list,x_mask,test_viz,name_to_sequence,OUTPUT)


        #then run through all my training iterations for those visualizations too
        for experiment_list in expr_list_of_lists_train:
            ascii_seq_list,x_vals,x_mask,y_vals = seqHelper.gen_batch_XY_rnn(experiment_list,name_to_sequence,\
                name_to_measurement,max_seq_length,alphabet_to_one_hot_sequence,output_dim)

            #run the training data again to get the visualization on what
            #   has already been trained
            test_prediction,test_error_output,test_viz = test_func(x_vals,x_mask,y_vals)

            #then write out the prediction for the training data
            seqHelper.write_out_rnn_prediction(experiment_list,x_mask,test_viz,name_to_sequence,OUTPUT)

        OUTPUT.close()

        #then write out my progress
        OUTPUT = open(progress_filename, 'a')
        OUTPUT.write(str(epoch)+","+str(np.sqrt(np.mean(testErrorList)))+','+str(np.mean(testErrorList))+'\n')
        OUTPUT.close()
