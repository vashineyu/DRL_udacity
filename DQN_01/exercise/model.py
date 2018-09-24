#import torch
#import torch.nn as nn
#import torch.nn.functional as F
import tensorflow as tf
import numpy as np

class QNetwork():
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, optimizer, seed, name):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state, example: (8,)
            action_size (int): Dimension of each action, example: 4
            seed (int): Random seed
            name: graph name
        """
        "*** YOUR CODE HERE ***"
        self.state_size = state_size
        self.action_size = action_size
        self.optim = optimizer
        self.graph_name = name

        print("State size: %i" % self.state_size)
        print("Action size: %i" % self.action_size)
        # Initalize
        tf.reset_default_graph()
        np.random.seed(seed)
        self.sess = tf.Session() # Prepare a tensorflow sesion
        self.input = tf.placeholder(shape = [None, self.state_size], dtype = tf.float32)
        self.output = tf.placeholder(shape = [None, self.action_size], dyype = tf.float32)

        # Build networks
        with tf.variable_scope("Qtable"):
            neurons_of_layers = [25, 25, 25]
            # Use to update/train the agent's brain
            self.q_local = self._build_model(neurons_of_layers = neurons_of_layers, scope = 'local', trainable = True)

            # Use to generate action (with fixed parameters)
            self.q_target = self._build_model(neurons_of_layers = neurons_of_layers, scope = 'target', trainable = False)
        
        # Handlers of parameters
        self.localnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Qtable/local')
        self.targetnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Qtable/target')
        self.params_replace = [tf.assign(old, new) for old, new in zip(self.localnet_params, self.targetnet_params)]

        # Compute loss
        self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.q_local))
        with tf.get_collection(tf.GraphKeys.UPDATE_OPS):
            self.update_ops = self.optim.minimize(self.loss, var_list = self.localnet_params)

        # Finally, initalize weights
        self.sess.run(tf.global_variables_initalizer())
        print("Network Ready")

    def _build_model(self, scope, trainable, neurons_of_layers = [25,25,25], acti = tf.nn.relu):
        """
        scope: local network or target network
        trainable: control if it is controlable or not
        neurons_of_layers: numbers of units for each layers
        acti: activation
        """
        # Defined block function, useful when repeats
        def mlp_block(x, name, units, activation, trainable = True):
            with tf.variable_scope(name):
                x = tf.layers.dense(x, units = units, trainable = trainable)
                x = activation(x)
            return x
            
        # Build a simple multi-layers network
        with tf.variable_scope(scope):
            for i, n_unit in enumerate(neurons_of_layers):
                block_name = 'block_' + str(i)
                # if i = 0 (first block), inputs should be input_placeholder
                x = mlp_block(x = self.input if i == 0 else x, 
                    name = block_name, 
                    units = n_unit, 
                    activation = acti, 
                    trainable = trainable)
            x = tf.layers.dense(x = x, units = self.action_size, trainable = trainable)
        return x

    def eval(self, state):
        """
        Generate action from targetnet
        """
        pass

    def train(self):
        """
        Train the localnet
        """
        pass

    def update_target_network(self):
        """
        Swap memory from localnet to targetnet,
        simply session run the replacement ops
        """
        self.sess.run(self.params_replace)

    def save_model(self):
        """
        Save the model
        """
        pass

    def load_model(self):
        """
        Load the model
        """
        pass








    
