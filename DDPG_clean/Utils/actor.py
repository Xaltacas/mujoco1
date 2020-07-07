#import tflearn
import tensorflow as tf
import numpy as np


class ActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, layers, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network(layers,"Actor")

        self.network_params = tf.compat.v1.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network(layers,"ActorTrack")

        self.target_network_params = tf.compat.v1.trainable_variables()[
                                     len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        with tf.compat.v1.variable_scope("ActorRegu"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate). \
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self, layers, name):

        inputs = tf.compat.v1.placeholder(tf.float32, (None, self.s_dim))
        with tf.compat.v1.variable_scope(name):
            layer_dims = [self.s_dim] + layers + [self.a_dim]
            x = inputs

            for i, (src_dim, tgt_dim) in enumerate(zip(layer_dims, layer_dims[1:])):
                Wi_name, bi_name = "W" +str(i), "b"+str(i)

                #Wi = ((track_scope and match_variable(Wi_name, track_scope))
                #      or tf.compat.v1.get_variable("W%i" % i, (src_dim, tgt_dim),initializer = tf.compat.v1.random_normal_initializer(stddev = 0.3)))
                #Wi = tf.Variable(tf.random.normal(shape=(src_dim, tgt_dim),stddev=0.5),name = Wi_name)
                Wi = tf.Variable(tf.random.normal(shape =(src_dim, tgt_dim)), name =  Wi_name)
                #Wi = tf.compat.v1.get_variable("W%i" % i, (src_dim, tgt_dim),initializer = tf.compat.v1.random_normal_initializer(stddev = 0.3))
                x = tf.matmul(x, Wi)

                bi = tf.Variable(np.zeros(shape = (tgt_dim,),dtype="float32"),name = bi_name)
                #bi = ((track_scope and match_variable(bi_name, track_scope))
                #    or tf.compat.v1.get_variable("b%i" % i, (tgt_dim,),initializer=tf.zeros_initializer))
                x += bi
                scaled_x = tf.nn.tanh(x)

                final_layer = i == len(layer_dims) - 2
                if not final_layer:
                    x = scaled_x

            scaled_x = tf.multiply(scaled_x, self.action_bound)
        return inputs, x, scaled_x



        #W1 = tf.Variable(tf.random.normal(shape =(self.s_dim, 400)))
        #net = tf.matmul(inputs, W1)
        #b1 = tf.Variable(np.zeros(shape = (400,),dtype="float32"))
        #net = net + b1
        #net = tf.nn.relu(net)


        #W2 = tf.Variable(tf.random.normal(shape=(400,300)))
        #net = tf.matmul(net, W2)
        #b2 = tf.Variable(np.zeros(shape =(300,),dtype="float32"))
        #net = net + b2
        #net = tf.nn.relu(net)


        #W3 = tf.Variable(np.float32(np.random.uniform(low=-0.003, high=0.003,size =(300,self.a_dim))))
        #out = tf.matmul(net,W3)
        #b3 = tf.Variable(np.zeros(shape =(self.a_dim,),dtype="float32"))
        #out = out + b3

        ## Scale output to -action_bound to action_bound
        #scaled_out = tf.nn.tanh(tf.multiply(out, self.action_bound))
        #return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
