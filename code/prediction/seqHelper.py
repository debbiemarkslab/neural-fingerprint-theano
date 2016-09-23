import numpy as np
import cPickle as pickle

#########################################
# Reads in the ECFP fingerprint as well as the values to train on
#   expr_filename : experiment filename to train on
#   fingerprint_filename : precomputed fingerprint filename
#
#   Returns : dictionaries of the smiles to prediction and smiles to fingerprint
#########################################
def read_in_ecfp_data(expr_filename, fingerprint_filename):
    smiles_to_prediction = {}
    smiles_to_fingerprint = {}

    #read in the experimental data
    INPUT = open(expr_filename, 'r')
    for line in INPUT:
        line = line.rstrip()
        line_list = line.split(',')
        pred = float(line_list[0])
        smiles = line_list[1]
        smiles_to_prediction[smiles] = pred
    INPUT.close()

    #read in the fingerprint data
    INPUT = open(fingerprint_filename, 'r')
    for line in INPUT:
        line = line.rstrip()
        line_list = line.split(',')
        smiles = line_list[0]
        fingerprint_list = [float(i) for i in list(line_list[1])]
        smiles_to_fingerprint[smiles] = np.asarray(fingerprint_list)
    INPUT.close()

    return smiles_to_prediction,smiles_to_fingerprint

#########################################
# Reads in the pickeled convolutional fingerprint as well as the values to train on
#   expr_filename : experiment filename to train on
#   fingerprint_feat_filename : precomputed fingerprint filename
#
#   Returns : dictionaries of the smiles to prediction and smiles to fingerprint
#########################################
def read_in_data(expr_filename,fingerprint_feat_filename):
    smiles_to_fingerprint_features = pickle.load(open(fingerprint_feat_filename, "rb" ))

    #first need to get the max atom length
    max_atom_len = 0
    max_bond_len = 0
    num_atom_features = 0
    num_bond_features = 0
    smiles_to_rdkit_list = {}

    for smiles,arrayrep in smiles_to_fingerprint_features.iteritems():
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        rdkit_list = arrayrep['rdkit_ix']
        smiles_to_rdkit_list[smiles] = rdkit_list

        atom_len,num_atom_features = atom_features.shape
        bond_len,num_bond_features = bond_features.shape

        if atom_len > max_atom_len:
            max_atom_len = atom_len
        if bond_len > max_bond_len:
            max_bond_len = bond_len

    #then add 1 so I can zero pad everything
    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len

    max_atom_len += 1
    max_bond_len += 1

    smiles_to_atom_info = {}
    smiles_to_bond_info = {}

    smiles_to_atom_neighbors = {}
    smiles_to_bond_neighbors = {}

    smiles_to_atom_mask = {}

    degrees = [0,1,2,3,4,5]
    #then run through our numpy array again
    for smiles,arrayrep in smiles_to_fingerprint_features.iteritems():
        mask = np.zeros((max_atom_len))

        #get the basic info of what
        #    my atoms and bonds are initialized
        atoms = np.zeros((max_atom_len,num_atom_features))
        bonds = np.zeros((max_bond_len,num_bond_features))

        #then get the arrays initlialized for the neighbors
        atom_neighbors = np.zeros((max_atom_len,len(degrees)))
        bond_neighbors = np.zeros((max_atom_len,len(degrees)))

        #now set these all to the last element of the list, which is zero padded
        atom_neighbors.fill(max_atom_index_num)
        bond_neighbors.fill(max_bond_index_num)

        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        for i,feature in enumerate(atom_features):
            mask[i] = 1.0
            atoms[i] = feature

        for j,feature in enumerate(bond_features):
            bonds[j] = feature

        atom_neighbor_count = 0
        bond_neighbor_count = 0
        working_atom_list = []
        working_bond_list = []
        for degree in degrees:
            atom_neighbors_list = arrayrep[('atom_neighbors', degree)]
            bond_neighbors_list = arrayrep[('bond_neighbors', degree)]

            if len(atom_neighbors_list) > 0:

                for i,degree_array in enumerate(atom_neighbors_list):
                    for j,value in enumerate(degree_array):
                        atom_neighbors[atom_neighbor_count,j] = value
                    atom_neighbor_count += 1

            if len(bond_neighbors_list) > 0:
                for i,degree_array in enumerate(bond_neighbors_list):
                    for j,value in enumerate(degree_array):
                        bond_neighbors[bond_neighbor_count,j] = value
                    bond_neighbor_count += 1

        #then add everything to my arrays
        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds

        smiles_to_atom_neighbors[smiles] = atom_neighbors
        smiles_to_bond_neighbors[smiles] = bond_neighbors

        smiles_to_atom_mask[smiles] = mask

    #now I don't need my pkl info
    del smiles_to_fingerprint_features

    smiles_to_measurement = {}
    INPUT = open(expr_filename, 'r')
    for line in INPUT:
        line = line.rstrip()
        line_list = line.split(',')
        measurement,smiles = line_list
        smiles_to_measurement[smiles] = float(measurement)
    INPUT.close()

    return smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
        smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask,\
        smiles_to_rdkit_list,max_atom_len,max_bond_len,num_atom_features,num_bond_features

######################################
#Creates random training and test sets
#    seq_list : list of smiles to train on
#    test_num : how many samples are desired in the test
#    random_seed : random seed to split data
#
#    Returns : lists of training and test data
######################################
def gen_rand_train_test_data(seq_list, test_num, random_seed):
	#first set the random seed
	np.random.seed(random_seed)
	train_list = []
	test_list = []
	#make a list of values I can test
	valList = np.arange(0,len(seq_list))
	#shuffle them
	np.random.shuffle(valList)
	#grab a list that is the size of my test num

	test_values = list(valList)[:test_num]
	train_values = list(valList)[test_num:]
	for number in test_values:
		test_list.append(seq_list[number])
	for number in train_values:
		train_list.append(seq_list[number])
	return test_list, train_list

######################################
#Generates random lists for minibatches
#    smiles_list : list of smiles to train on
#    batch_size : size of batch
#    random_seed : random seed to split data
#
#    Returns : list of lists that are each a minibatch
######################################
def gen_batch_list_of_lists(smiles_list,batch_size,random_seed):
	np.random.seed(random_seed)
	batch_list_of_lists = []
	#make them a numpy list so I can shuffle them
	smiles_for_shuffle = np.array(smiles_list[:])
	#shuffle them
	np.random.shuffle(smiles_for_shuffle)

	smiles_list_for_shuffle = smiles_for_shuffle.tolist()
	for i in xrange(0,len(smiles_list),batch_size):
		batch_list_of_lists.append(smiles_list_for_shuffle[i:i+batch_size])
	return batch_list_of_lists

####################################################
#Generates minitbatches of the CNN data
#
####################################################
def gen_batch_XY_reg(experiment_list,smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
    smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask):

    y_val = []
    x_mask = []
    x_atom = []
    x_bonds = []
    x_atom_index = []
    x_bond_index = []
    for smiles in experiment_list:
        y_val.append(smiles_to_measurement[smiles])
        x_mask.append(smiles_to_atom_mask[smiles])
        x_atom.append(smiles_to_atom_info[smiles])
        x_bonds.append(smiles_to_bond_info[smiles])
        x_atom_index.append(smiles_to_atom_neighbors[smiles])
        x_bond_index.append(smiles_to_bond_neighbors[smiles])
    return np.asarray(x_atom),np.asarray(x_bonds),np.asarray(x_atom_index),\
        np.asarray(x_bond_index),np.asarray(x_mask),np.asarray(y_val)

################################################
#Makes a dictionary that turns the alphabet
#   name_to_sequence : experiment name to sequences
#
#   Returns:
#       alphabet_to_ordinal : dictionary of alphabet letter -> ordinal number
#                                indexing for one-hot matrix
#       ordinal_to_alphabet : dictionary of ordinal number -> alphabet letter
################################################
def make_one_hot_dict(name_to_sequence):
    alphabet = []
    #was getting strange errors with some sequences; now default behavior
    if len(name_to_sequence) > 100000000000:
        counter = 0
        for name,sequence in name_to_sequence.iteritems():
            counter += 1
            for letter in sequence:
                if letter not in alphabet:
                    alphabet.append(letter)
            if counter > 1000:
                break
    else:
        for name,sequence in name_to_sequence.iteritems():
            for letter in sequence:
                if letter not in alphabet:
                    alphabet.append(letter)

    alphabet_to_ordinal = {}
    ordinal_to_alphabet = {}
    for i,letter in enumerate(sorted(alphabet)):
        alphabet_to_ordinal[letter] = i
        ordinal_to_alphabet[i] = letter

    return alphabet_to_ordinal, ordinal_to_alphabet

##############################################
#Reads in the sequences for the RNN prediction as well as the y output
#    filename : test filename
#
#    Returns : dictionaries containing:
#                   expr num -> sequence
#                   smiles -> y output
#                   the maximum sequence length
##############################################
def read_in_chem_seq(filename):
    name_to_sequence = {}
    name_to_chem_val = {}
    INPUT = open(filename, 'r')
    max_seq_length = 0
    for iterator,line in enumerate(INPUT):
        line = line.rstrip()
        line_list = line.split(',')

        smiles = line_list[1]

        name_to_sequence[iterator]= smiles

        #check the length of the longest sequence
        #we need to get this length to define our tensor
        if len(line_list[1]) > max_seq_length:
            max_seq_length = len(line_list[1])

        name_to_chem_val[iterator]=float(line_list[0])
    INPUT.close()
    return name_to_sequence,name_to_chem_val,max_seq_length


###############################################
#Writes out a csv file that holds the visualizations from the visual CNN
#   experiment_list : list of smiles used
#   x_mask : sequence mask
#   smiles_to_rdkit_list : dictionary of smiles -> rdkit representation/count of atoms and bonds
#   test_viz : list containing  the data needed for visualizations
#   test_prediction : predicted output values
#   OUTPUTVIZ : fileobject to write out visualizations
#
#   Returns : None
##############################################
def write_out_rnn_prediction(experiment_list,x_mask,test_viz,test_prediction,name_to_sequence,OUTPUT):

    for i,expr_number in enumerate(experiment_list):
        ind_mask = x_mask[i]
        ind_pred = test_viz[i]
        smiles = name_to_sequence[expr_number]
        OUTPUT.write(smiles+','+str(test_prediction[i])+',')
        out_pred = []
        for j,one_or_zero in enumerate(list(ind_mask)):
            if one_or_zero == 1.0:
                out_pred.append(str(ind_pred[j][0]))

            OUTPUT.write(":".join(out_pred)+'\n')

###################################################
#Generates the minibatch used for training
#    experiment_list : smiles used for this training iteration
#    smiles_to_prediction : dictionary from smiles -> y value of prediction
#    smiles_to_fingerprint : dictionary from smiles -> fingerprint
#
#    Returns: minitbatches of above data
###################################################
def gen_batch_XY_rnn(seq_list,name_to_sequence,name_to_chem_val,max_seq_length, alphabet_to_one_hot_sequence,output_dim):
    #initialize my x_vals, y_vals, and x_mask to zeros so I can add ones to them later
    x_vals = np.zeros((len(seq_list),max_seq_length,len(alphabet_to_one_hot_sequence)), dtype=np.float32)
    y_vals = np.zeros((len(seq_list)),dtype=np.float32)
    x_mask = np.zeros((len(seq_list),max_seq_length))
    asciiseq_list = []
    for i,seqName in enumerate(seq_list):
        sequence = name_to_sequence[seqName]
        asciiseq_list.append(sequence)
        #to the x val of the sequence
        for j,letter in enumerate(sequence):
            k = alphabet_to_one_hot_sequence[letter]
            x_vals[i,j,k] = 1.0
            x_mask[i,j] = 1.0
        y_vals[i] = name_to_chem_val[seqName]
    return asciiseq_list,x_vals,x_mask,y_vals


##############################################
#Writes out a csv file that holds the visualizations from the visual CNN
#   experiment_list : list of smiles used
#   x_mask : sequence mask
#   smiles_to_rdkit_list : dictionary of smiles -> rdkit representation/count of atoms and bonds
#   vis_list : list containing  the data needed for visualizations
#   test_prediction : predicted output values
#   OUTPUTVIZ : fileobject to write out visualizations
#
#   Returns : None
##############################################
def write_out_predictions_cnn(experiment_list, x_mask, smiles_to_rdkit_list,
    viz_list, test_prediction, OUTPUTVIZ):

    for i,smiles in enumerate(experiment_list):
        ind_mask = x_mask[i]
        ind_atom = smiles_to_rdkit_list[smiles]
        ind_pred = viz_list[i]

        out_atom = []
        out_pred = []
        out_atom_features = []
        OUTPUTVIZ.write(smiles+','+str(test_prediction[i])+',')
        for j,one_or_zero in enumerate(list(ind_mask)):
            if one_or_zero == 1.0:
                out_atom.append(str(ind_atom[j]))
                out_pred.append(str(ind_pred[j][0]))

        OUTPUTVIZ.write(":".join(out_atom)+','+":".join(out_pred)+'\n')

###################################################
#Generates the minibatch used for training
#    experiment_list : smiles used for this training iteration
#    smiles_to_prediction : dictionary from smiles -> y value of prediction
#    smiles_to_fingerprint : dictionary from smiles -> fingerprint
#
#    Returns: minitbatches of above data
###################################################
def gen_batch_XY_control(experiment_list, smiles_to_prediction, \
    smiles_to_fingerprint):

    x_fing_list = []
    y_val_list = []
    for smiles in experiment_list:
        x_fing_list.append(smiles_to_fingerprint[smiles])
        y_val_list.append(smiles_to_prediction[smiles])

    return np.asarray(x_fing_list),np.asarray(y_val_list)
