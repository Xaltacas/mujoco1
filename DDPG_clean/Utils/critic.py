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

        # Op for periodically updating target network with online network weights with regularization
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

        inputs = tf.compat.v1.placeholder(tf.float32, (None, self.s_dim))
        action = tf.compat.v1.placeholder(tf.float32, (None, self.a_dim))

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

                Wix = tf.Variable(tf.random.normal(shape =(src_dim, tgt_dim)), name =  Wi_name+"_x")
                Wia = tf.Variable(tf.random.normal(shape =(src_dim, tgt_dim)), name =  Wi_name+"_a")

                x = tf.matmul(x, Wix)
                a = tf.matmul(a, Wia)

                bix = tf.Variable(np.zeros(shape = (tgt_dim,),dtype="float32"),name = bi_name+"_x")
                bia = tf.Variable(np.zeros(shape = (tgt_dim,),dtype="float32"),name = bi_name+"_a")
                x += bix
                a += bia


            out = tf.nn.relu(a+x)
            Wout = tf.Variable(np.float32(np.random.uniform(low=-0.003, high=0.003, size =(layers[-1], 1))), name =  "Wout")
            out = tf.matmul(out,Wout)
            bout = tf.Variable(np.zeros(shape = (1,),dtype="float32"),name = "bout")
            out = out + bout


        return inputs, action, out

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
