import lasagne
from lasagne.layers.shape import PadLayer
from lasagne.layers.merge import ConcatLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, ReshapeLayer, dropout
from lasagne.nonlinearities import softmax, rectify
#from lasagne.layers.normalization import BatchNormLayer, batch_norm
from lasagne.layers.base import Layer
from lasagne import nonlinearities
from lasagne.layers.merge import MergeLayer
#import bayesianRNNlasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng
from lasagne.utils import unroll_scan
import theano
import numpy as np
from theano.tensor import *
import theano.tensor as T
from theano.tensor.nnet import conv3d2d
from lasagne import init
from lasagne import nonlinearities




class VizAndPredMolLayer(MergeLayer):
    def __init__(self, incoming, num_input_units, num_output_units,
        nonlinearity=lasagne.nonlinearities.linear,num_hidden_units=0,**kwargs):
        super(VizAndPredMolLayer, self).__init__(incoming, **kwargs)

        W = lasagne.init.GlorotUniform()
        b = lasagne.init.Constant(0.)
        self.nonlinearity = nonlinearity
        self.num_input_units = num_input_units
        self.num_output_units = num_output_units

        self.num_hidden_units = num_hidden_units

        if num_hidden_units == 0:
            self.W = self.add_param(W,(num_input_units,num_output_units),name="W_proj")
            self.b = self.add_param(b,(num_output_units,),name="b_proj",regularizable=False)

        else:
            self.W_1 = self.add_param(W,(num_input_units,num_hidden_units),name="W_1")
            self.b_1 = self.add_param(b,(num_hidden_units,),name="b_1",regularizable=False)

            self.W_2 = self.add_param(W,(num_hidden_units,num_output_units),name="W_2")
            self.b_2 = self.add_param(b,(num_output_units,),name="b_2",regularizable=False)


    def get_output_shape_for(self, input_shapes):
        #this returns the shape of the concatenated layer
        #dimensionality: (batch_size,sequence_length,hidden_units_num)
        input_shape = input_shapes[0]
        return (input_shape[0],self.num_output_units)

    def get_output_for(self, inputs, **kwargs):
        #concat is the first input
        fingerprints = inputs[0]
        mask = inputs[1]

        batch_size,seq_len,fingerprint_dim = fingerprints.shape

        #reshape them so I can do the multiplcation quickly
        fingerprints = fingerprints.reshape((batch_size*seq_len,fingerprint_dim))

        if self.num_hidden_units == 0:
            f_out = T.dot(fingerprints,self.W)+self.b
        else:
            f_mid = lasagne.nonlinearities.rectify(T.dot(fingerprints,self.W_1)+self.b_1)
            f_out = T.dot(f_mid,self.W_2)+self.b_2


        atom_prediction = f_out.reshape((batch_size,seq_len,self.num_output_units)) \
            * mask.dimshuffle(0,1,'x')
        molecule_prediction = T.sum(atom_prediction,axis=1)
        return molecule_prediction, atom_prediction






class FingerprintGen(MergeLayer):
    def __init__(self, incoming, **kwargs):
        super(FingerprintGen, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shapes):
        #this returns the shape of the concatenated layer
        #dimensionality: (batch_size,sequence_length,hidden_units_num)
        input_shape = input_shapes[0]
        return (input_shape[0],input_shape[2])

    def get_output_for(self, inputs, **kwargs):
        #concat is the first input
        fingerprint_proj = inputs[0]
        mask = inputs[1]

        fingerprint_mask = (mask.dimshuffle(0, 1, 'x') * fingerprint_proj)
        summed_vals = T.sum(fingerprint_mask,axis=1)

        return summed_vals



class FingerprintGenTopSum(MergeLayer):
    def __init__(self, incoming, **kwargs):
        super(FingerprintGenTopSum, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shapes):
        #this returns the shape of the concatenated layer
        #dimensionality: (batch_size,sequence_length,hidden_units_num)
        input_shape = input_shapes[0]
        return input_shape

    def get_output_for(self, inputs, **kwargs):
        current_fingerprint = inputs[0]

        new_fingerprint = inputs[1]

        summed_vals = T.sum([current_fingerprint,new_fingerprint],axis=0)
        return summed_vals



class SparsifyFingerprintLayer(MergeLayer):

    def __init__(self, incoming, num_mol_units, num_proj_units,\
        W_proj=lasagne.init.GlorotUniform(),b_proj=lasagne.init.Constant(0.),\
        **kwargs):
        super(SparsifyFingerprintLayer, self).__init__(incoming, **kwargs)

        #initlialize the values of the weight matrices I'm going to use to transform
        self.W_proj_mol = self.add_param(W_proj,(num_mol_units,num_proj_units),name="W_proj_mol")
        self.b_proj = self.add_param(b_proj,(num_proj_units,),name="b_proj",regularizable=False)

        self.num_mol_units = num_mol_units
        self.num_proj_units = num_proj_units


    def get_output_shape_for(self,input_shapes):
        #the input shapes are [(batchSize,seqLengthAttn,input_feature_num)], -> l_features

        #get the shape of the atom list
        batch_size,length,input_units = input_shapes[0]

        #then use the (batchSize,seqLength,hidden_units_num)
        self.output_shapes = (batch_size,length,self.num_proj_units)

        return self.output_shapes

    def get_output_for(self, inputs, deterministic=False, **kwargs):


        #these are the features that were input
        mol_features = inputs[0]

        #reshape them to I can multiply along the length of the molecule
        batch_num,length,num_units = mol_features.shape
        mol_features_reshape = mol_features.reshape((batch_num*length,num_units))

        mol_proj_flat = T.dot(mol_features_reshape,self.W_proj_mol)

        mol_proj = mol_proj_flat.reshape((batch_num,length,self.num_proj_units))

        unnormalized_proj = mol_proj + self.b_proj

        e_x = T.exp(unnormalized_proj - unnormalized_proj.max(axis=2, keepdims=True))
        sparse_proj_features = e_x / e_x.sum(axis=2, keepdims=True)

        return sparse_proj_features


class FingerprintHiddensLayer(MergeLayer):
    def __init__(self, incoming, input_feature_num, input_atom_num, input_bond_num,
        hidden_units_num, max_atom_len, p_dropout=0.0,\
        W_neighbors=lasagne.init.GlorotUniform(),b_neighbors=lasagne.init.Constant(0.),\
        W_atoms=lasagne.init.GlorotUniform(),b_atoms=lasagne.init.Constant(0.),\
        nonlinearity=nonlinearities.rectify,batch_normalization=True, **kwargs):
        super(FingerprintHiddensLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        #initlialize the values of the weight matrices I'm going to use to transform
        self.W_neighbors = self.add_param(W_neighbors,(input_feature_num+input_bond_num,hidden_units_num),name="W_neighbors")
        self.b_neighbors = self.add_param(b_neighbors,(hidden_units_num,),name="b_neighbors",regularizable=False)

        self.W_atoms = self.add_param(W_atoms,(input_atom_num,hidden_units_num),name="W_atoms")
        self.b_atoms = self.add_param(b_atoms,(hidden_units_num,),name="b_atoms",regularizable=False)

        self.num_units = hidden_units_num
        self.atom_num = input_atom_num
        self.input_feature_num = input_feature_num

        self.p_dropout = p_dropout
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.length = max_atom_len


    def get_output_shape_for(self,input_shapes):
        #the input shapes are [(batchSize,seqLengthAttn,input_feature_num)], -> l_features

        #get the shape of the atom list
        batch_size,length,input_units = input_shapes[1]

        #then use the (batchSize,seqLength,hidden_units_num)
        self.output_shapes = (batch_size,length,self.num_units)

        return self.output_shapes

    def get_output_for(self, inputs, deterministic=False, **kwargs):


        #first unpack our inputs
        #first we want our mask of the atoms
        atom_mask = inputs[5]

        #unpack our inputs
        atom_degree_list = inputs[0]
        bond_degree_list = inputs[1]

        #these our the basic atoms and bonds we get
        atom_list = inputs[2]
        bond_list = inputs[3]

        #these are the features that were input
        feature_list = inputs[4] * atom_mask.dimshuffle(0,1,'x')


        #reshape my atoms so I can do this all at once
        batch_size,mol_length,num_atom_feat = atom_list.shape
        atom_list_reshape = atom_list.reshape((batch_size*mol_length,num_atom_feat))

        #transform my atoms
        atom_transformed_flat = T.dot(atom_list_reshape,self.W_atoms) + self.b_atoms
        atom_transformed = atom_transformed_flat.reshape((batch_size,mol_length,self.num_units))

        ##########################################
        #To index into our batches to grab the atom and bond neighbors,
        #    we need to loop through the first dimension of the batches to
        #    get our neighbors before we can do mulitplication
        #This can be done with the map function
        ############################################
        iteration_list = T.arange(batch_size)

        def index_bonds(i,bond_list,bond_degree_list,*args):
            return bond_list[i][bond_degree_list[i]]

        def index_features(i,feature_list,atom_degree_list,*args):
            return feature_list[i][atom_degree_list[i]]

        bonds_indexed = theano.map(fn=index_bonds,
            sequences=iteration_list,
            non_sequences=[bond_list,bond_degree_list])[0]

        features_indexed = theano.map(fn=index_features,
            sequences=iteration_list,
            non_sequences=[feature_list,atom_degree_list])[0]

        #stack the features and bonds
        stacked_features = T.concatenate([features_indexed,bonds_indexed],axis=-1)
        #then sum them
        summed_features = T.sum(stacked_features,axis=2)

        #reshape my atoms so I can do matrix multipication all at once on the gpu
        batch_size,mol_length,num_summed_feat = summed_features.shape
        summed_features_list_reshape = summed_features.reshape((batch_size*mol_length,num_summed_feat))

        #transform my atoms
        summed_features_transformed_flat = T.dot(summed_features_list_reshape,self.W_neighbors) + self.b_neighbors
        weighted_features = summed_features_transformed_flat.reshape((batch_size,mol_length,self.num_units))

        #sum the atoms and bonds
        atoms_and_features = weighted_features + atom_transformed

        #get the mean and variance
        batch_mean = T.mean(atoms_and_features,axis=1,keepdims=True)
        batch_var = T.var(atoms_and_features,axis=1,keepdims=True)

        #then normalize
        normed_features = (atoms_and_features - batch_mean) / \
            T.sqrt(batch_var + 0.00001)

        #do nonlinearity
        activated_features = self.nonlinearity(normed_features)

        return activated_features



class FingerprintMerge(MergeLayer):

    def __init__(self, incoming, **kwargs):
        super(FingerprintMerge, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shapes):
        #this returns the shape of the concatenated layer
        #dimensionality: (batch_size,sequence_length,hidden_units_num)
        input_shape = input_shapes[0]
        return input_shape

    def get_output_for(self, inputs, **kwargs):
        #concat is the first input
        old_fingerprints = inputs[0]
        #reweighting is the second input
        new_fingerprints = inputs[1]
        #multiply the reweighting by the concatenation to get the scaling
        return (old_fingerprints + new_fingerprints)


class Gate(object):
    """
    lasagne.layers.recurrent.Gate(W_in=lasagne.init.Normal(0.1),
    W_hid=lasagne.init.Normal(0.1), W_cell=lasagne.init.Normal(0.1),
    b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.sigmoid)
    Simple class to hold the parameters for a gate connection.  We define
    a gate loosely as something which computes the linear mix of two inputs,
    optionally computes an element-wise product with a third, adds a bias, and
    applies a nonlinearity.
    Parameters
    ----------
    W_in : Theano shared variable, numpy array or callable
        Initializer for input-to-gate weight matrix.
    W_hid : Theano shared variable, numpy array or callable
        Initializer for hidden-to-gate weight matrix.
    W_cell : Theano shared variable, numpy array, callable, or None
        Initializer for cell-to-gate weight vector.  If None, no cell-to-gate
        weight vector will be stored.
    b : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector.
    nonlinearity : callable or None
        The nonlinearity that is applied to the input gate activation. If None
        is provided, no nonlinearity will be applied.
    Examples
    --------
    For :class:`LSTMLayer` the bias of the forget gate is often initialized to
    a large positive value to encourage the layer initially remember the cell
    value, see e.g. [1]_ page 15.
    >>> import lasagne
    >>> forget_gate = Gate(b=lasagne.init.Constant(5.0))
    >>> l_lstm = LSTMLayer((10, 20, 30), num_units=10,
    ...                    forgetgate=forget_gate)
    References
    ----------
    .. [1] Gers, Felix A., Jurgen Schmidhuber, and Fred Cummins. "Learning to
           forget: Continual prediction with LSTM." Neural computation 12.10
           (2000): 2451-2471.
    """
    def __init__(self, W_in=init.Normal(0.1), W_hid=init.Normal(0.1),
                 W_cell=init.Normal(0.1), b=init.Constant(0.),
                 nonlinearity=nonlinearities.sigmoid):
        self.W_in = W_in
        self.W_hid = W_hid
        # Don't store a cell weight vector when cell is None
        if W_cell is not None:
            self.W_cell = W_cell
        self.b = b
        # For the nonlinearity, if None is supplied, use identity
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity


class LSTMLayer(MergeLayer):
    r"""
    lasagne.layers.recurrent.LSTMLayer(incoming, num_units,
    ingate=lasagne.layers.Gate(), forgetgate=lasagne.layers.Gate(),
    cell=lasagne.layers.Gate(
    W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
    outgate=lasagne.layers.Gate(),
    nonlinearity=lasagne.nonlinearities.tanh,
    cell_init=lasagne.init.Constant(0.),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    peepholes=True, gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    A long short-term memory (LSTM) layer.
    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by
    .. math ::
        i_t &= \sigma_i(x_t W_{xi} + h_{t-1} W_{hi}
               + w_{ci} \odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(x_t W_{xf} + h_{t-1} W_{hf}
               + w_{cf} \odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t \odot \sigma_c(x_t W_{xc} + h_{t-1} W_{hc} + b_c)\\
        o_t &= \sigma_o(x_t W_{xo} + h_{t-1} W_{ho} + w_{co} \odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    ingate : Gate
        Parameters for the input gate (:math:`i_t`): :math:`W_{xi}`,
        :math:`W_{hi}`, :math:`w_{ci}`, :math:`b_i`, and :math:`\sigma_i`.
    forgetgate : Gate
        Parameters for the forget gate (:math:`f_t`): :math:`W_{xf}`,
        :math:`W_{hf}`, :math:`w_{cf}`, :math:`b_f`, and :math:`\sigma_f`.
    cell : Gate
        Parameters for the cell computation (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    outgate : Gate
        Parameters for the output gate (:math:`o_t`): :math:`W_{xo}`,
        :math:`W_{ho}`, :math:`w_{co}`, :math:`b_o`, and :math:`\sigma_o`.
    nonlinearity : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial cell state (:math:`c_0`).
    hid_init : callable, np.ndarray, theano.shared or :class:`Layer`
        Initializer for initial hidden state (:math:`h_0`).
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `ingate.W_cell`, `forgetgate.W_cell` and
        `outgate.W_cell` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(LSTMLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, num_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (num_units, num_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``. When the hidden state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When the cell state of this layer is to be
            pre-filled (i.e. was set to a :class:`Layer` instance) `inputs`
            should have length at least 2, and `inputs[-1]` is the hidden state
            to prefill with. When both the cell state and the hidden state are
            being pre-filled `inputs[-2]` is the hidden state, while
            `inputs[-1]` is the cell state.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out
