#import tflearn
import tensorflow as tf
import numpy as np


class CriticNetwork(object):

    def __init__(self, sess, state_dim, action_dim, layers, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network(layers,"Critic")

        self.network_params = tf.compat.v1.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network(layers,"CriticTrack")

        self.target_network_params = tf.compat.v1.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        with tf.compat.v1.variable_scope("CriticRegu"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
                                                      + tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self, layers, name):


        #inputs = tflearn.input_data(shape=[None, self.s_dim])
        inputs = tf.compat.v1.placeholder(tf.float32, (None, self.s_dim))
        #action = tflearn.input_data(shape=[None, self.a_dim])
        action = tf.compat.v1.placeholder(tf.float32, (None, self.a_dim))
        #net = tflearn.fully_connected(inputs, 400)

        with tf.compat.v1.variable_scope(name):
            layer_dims = layers
            x = inputs
            a = action
            Wix = tf.Variable(tf.random.normal(shape =(self.s_dim, layers[0])), name =  "W0_x")
            Wia = tf.Variable(tf.random.normal(shape =(self.a_dim, layers[0])), name =  "W0_a")
            x = tf.matmul(x, Wix)
            a = tf.matmul(a, Wia)
            bix = tf.Variable(np.zeros(shape = (layers[0],),dtype="float32"),name = "b0_x")
            bia = tf.Variable(np.zeros(shape = (layers[0],),dtype="float32"),name = "b0_a")
            x += bix
            a += bia



            for i, (src_dim, tgt_dim) in enumerate(zip(layer_dims, layer_dims[1:])):
                Wi_name, bi_name = "W" +str(i+1), "b"+str(i+1)

                x = tf.nn.relu(x)
                a = tf.nn.relu(a)

                #Wi = ((track_scope and match_variable(Wi_name, track_scope))
                #      or tf.compat.v1.get_variable("W%i" % i, (src_dim, tgt_dim),initializer = tf.compat.v1.random_normal_initializer(stddev = 0.3)))
                #Wi = tf.Variable(tf.random.normal(shape=(src_dim, tgt_dim),stddev=0.5),name = Wi_name)
                Wix = tf.Variable(tf.random.normal(shape =(src_dim, tgt_dim)), name =  Wi_name+"_x")
                Wia = tf.Variable(tf.random.normal(shape =(src_dim, tgt_dim)), name =  Wi_name+"_a")

                #Wi = tf.compat.v1.get_variable("W%i" % i, (src_dim, tgt_dim),initializer = tf.compat.v1.random_normal_initializer(stddev = 0.3))
                x = tf.matmul(x, Wix)
                a = tf.matmul(a, Wia)

                bix = tf.Variable(np.zeros(shape = (tgt_dim,),dtype="float32"),name = bi_name+"_x")
                bia = tf.Variable(np.zeros(shape = (tgt_dim,),dtype="float32"),name = bi_name+"_a")
                #bi = ((track_scope and match_variable(bi_name, track_scope))
                #    or tf.compat.v1.get_variable("b%i" % i, (tgt_dim,),initializer=tf.zeros_initializer))
                x += bix
                a += bia


            out = tf.nn.relu(a+x)
            Wout = tf.Variable(np.float32(np.random.uniform(low=-0.003, high=0.003, size =(layers[-1], 1))), name =  "Wout")
            out = tf.matmul(out,Wout)
            bout = tf.Variable(np.zeros(shape = (1,),dtype="float32"),name = "bout")
            out = out + bout


        return inputs, action, out


        #W1 = tf.Variable(tf.random.normal(shape =(self.s_dim, 400)))
        #net = tf.matmul(inputs, W1)
        #b1 = tf.Variable(np.zeros(shape = (400,),dtype="float32"))
        #net = net + b1
        ##net = tflearn.layers.normalization.batch_normalization(net)
        ##net = tflearn.activations.relu(net)
        #net = tf.nn.relu(net)

        ## Add the action tensor in the 2nd hidden layer
        ## Use two temp layers to get the corresponding weights and biases
        ##t1 = tflearn.fully_connected(net, 300)
        #W2 = tf.Variable(tf.random.normal(shape =(400 , 300)))
        #t1 = tf.matmul(net,W2)

        ##t2 = tflearn.fully_connected(action, 300)
        #W3 = tf.Variable(tf.random.normal(shape =(self.a_dim, 300)))
        #t2 = tf.matmul(action,W3)
        #b3 = tf.Variable(np.zeros(shape = (300,),dtype="float32"))
        #t2 = t2 + b3

        #net = tf.nn.relu(t1 + t2)

        ## linear layer connected to 1 output representing Q(s,a)
        ## Weights are init to Uniform[-3e-3, 3e-3]
        ##w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        ##out = tflearn.fully_connected(net, 1, weights_init=w_init)
        #W4 = tf.Variable(np.float32(np.random.uniform(low=-0.003, high=0.003,size =(300,1))))
        #out = tf.matmul(net,W4)
        #b4 = tf.Variable(np.zeros(shape = (1,),dtype="float32"))
        #out = out + b4

        #return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)