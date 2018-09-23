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
        np.random.seed(seed)
        self.build_model()

    def build_model(self, numbers_of_mlp_blocks = 3):

        def mlp_block(x, name, units = 25, activation = tf.nn.relu):
            with tf.variable_scope(name):
                x = tf.layers.dense(x, units = units)
                x = activation(x)
            return x

        with tf.variable_scope(self.graph_name):
            # Define inputs and outpts
            input_ = tf.placeholder(shape = [None, self.state_size], dtype = tf.float32)
            output_ = tf.placeholder(shape = [None, self.action_size], dyype = tf.float32)

            # Build a simple multi-layers network

    def eval(self, state):
        pass

    def train(self):
        pass

    def get_network_weights(self):
        pass








    
