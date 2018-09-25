import tensorflow as tf
import numpy as np

class QNetwork():
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, optimizer, gamma = 0.9, seed = 42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state, example: (8,)
            action_size (int): Dimension of each action, example: 4
            seed (int): Random seed
        """
        "*** YOUR CODE HERE ***"
        self.state_size = state_size
        self.action_size = action_size
        self.optim = optimizer
        self.gamma = gamma

        print("State size: %i" % self.state_size)
        print("Action size: %i" % self.action_size)
        # Initalize
        tf.reset_default_graph()
        np.random.seed(seed)
        self.sess = tf.Session() # Prepare a tensorflow sesion

        # Prepared placeholders
        self.state = tf.placeholder(shape = [None, self.state_size], dtype = tf.float32) # input: S
        self.next_state = tf.placeholder(shape = [None, self.state_size], dtype = tf.float32) # input: S'
        self.action = tf.placeholder(shape = [None, 1], dtype = tf.int32) # action
        self.reward = tf.placeholder(shape = [None, 1], dtype = tf.float32) # reward
        self.done = tf.placeholder(shape = [None, 1], dtype = tf.float32) # done or not

        self.action_onehot = tf.one_hot(self.action, depth = action_size)

        # Build networks
        with tf.variable_scope("Qtable"):
            neurons_of_layers = [64, 64, 64]
            # Use to update/train the agent's brain
            # Used to get Q(s, a)
            self.q_local = self._build_model(x = self.state, 
                                             neurons_of_layers = neurons_of_layers, 
                                             scope = 'local', trainable = True)

            # with fixed parameters, used to get Q(s', a)
            self.q_target = self._build_model(x = self.next_state, 
                                              neurons_of_layers = neurons_of_layers, 
                                              scope = 'target', trainable = False)
        
        # Handlers of parameters
        self.localnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Qtable/local')
        self.targetnet_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Qtable/target')
        self.params_replace = [tf.assign(old, new) for old, new in zip(self.localnet_params, self.targetnet_params)]

        # Compute loss (TD-loss), TD_error = lr * ((reward + gamma * max(Q(s',A)) - Q(s,a))
        td_target = self.reward + self.gamma * tf.reduce_max(self.q_target, axis = -1) * (1. - self.done )
        td_expect = tf.reduce_sum(self.q_local*self.action_onehot, axis = -1)
        self.loss = tf.reduce_mean(tf.squared_difference(td_target, td_expect))
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.update_ops = self.optim.minimize(self.loss, var_list = self.localnet_params)

        # Finally, initalize weights
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        print("Network Ready")

    def _build_model(self, x, scope, trainable, neurons_of_layers = [25,25,25], acti = tf.nn.relu):
        """
        x: input_placeholder
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
                x = mlp_block(x = x, 
                    name = block_name, 
                    units = n_unit, 
                    activation = acti, 
                    trainable = trainable)
            x = tf.layers.dense(x, units = self.action_size, trainable = trainable)
        return x

    def get_action(self, state):
        """
        Generate action from localnet
        get the action with the highest state-action pair
        """
        action = np.argmax(self.sess.run(self.q_local, feed_dict = {self.state:state}))
        return action

    def train(self, batch):
        """
        Train the localnet
        - Args:
            - batch: (state, action, reward, next_state, done)
        - Returns:
            - loss
        """
        state, action, reward, next_state, done = batch
        current_loss, _ = self.sess.run([self.loss, self.update_ops], feed_dict= {self.state:state,
                                                                                  self.action: action,
                                                                                  self.reward: reward,
                                                                                  self.next_state: next_state,
                                                                                  self.done: done})
        return current_loss

    def update_target_network(self):
        """
        Swap memory from localnet to targetnet,
        simply session run the replacement ops
        """
        self.sess.run(self.params_replace)

    def save_model(self, model_name = None):
        """
        Save the model (save localnet, we don't save target net)
        """
        saver.save(self.sess, 'dqn.ckpt' if model_name is None else model_name)
        print("model saved")

    def load_model(self, model_name = None):
        """
        Load the model
        """
        saver.restore(self.sess, 'dqn.ckpt' if model_name is None else model_name)
        print("model loaded")
