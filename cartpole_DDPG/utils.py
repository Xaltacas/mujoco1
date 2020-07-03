import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Buffer:
    def __init__(self):
        self.s = []
        self.a = []
        self.r = []
        self.s1 = []
        self.size = 0

    def extend(self,s,a,r,s1):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s1.append(s1)
        self.size += 1

    def sample(self,size):

        if self.size < size:
            raise ValueError("Not enough examples in buffer (just %i) to fill a batch of %i."
               % (self.size, size))

        idxs = np.random.choice(range(self.size), size, replace=False)
        #print("======")
        #print(idxs)
        #print("======")
        idxs = idxs.astype(int)
        return np.squeeze(np.array(self.s)[idxs]), np.squeeze(np.array(self.a)[idxs]), np.squeeze(np.array(self.r)[idxs]), np.squeeze(np.array(self.s1)[idxs])

class ReplayBuffer(object):

  """
  Experience replay storage, defined relative to an MDP.
  Stores experience tuples `(s_t, a_t, r_t, s_{t+1})` in a fixed-size cyclic
  buffer and randomly samples tuples from this buffer on demand.
  """

  def __init__(self, buffer_size, state_dim, action_dim):
    self.buffer_size = buffer_size
    self.state_dim = state_dim
    self.action_dim = action_dim

    self.cursor_write_start = 0
    self.cursor_read_end = 0

    self.states = np.empty((buffer_size, self.state_dim), dtype=np.float32)
    self.actions = np.empty((buffer_size,self.action_dim), dtype=np.int32)
    self.rewards = np.empty((buffer_size,), dtype=np.int32)
    self.states_next = np.empty_like(self.states)

  def sample(self, batch_size):
    if self.cursor_read_end - 1 < batch_size:
      raise ValueError("Not enough examples in buffer (just %i) to fill a batch of %i."
               % (self.cursor_read_end, batch_size))

    idxs = np.random.choice(self.cursor_read_end, size=batch_size, replace=False)
    return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
            self.states_next[idxs])

  def extend(self, states, actions, rewards, states_next):
    # If the buffer is near full, fit what we can and drop the rest
    remaining_space = self.buffer_size - self.cursor_write_start
    if len(states) >= self.buffer_size - self.cursor_write_start:
      states = states[:remaining_space]
      actions = actions[:remaining_space]
      rewards = rewards[:remaining_space]
      states_next = states_next[:remaining_space]

      # Reset for next time
      self.cursor_write_start = 0

    # Write into buffer.
    self.states[self.cursor_write_start:len(states)] = states
    self.actions[self.cursor_write_start:len(states)] = actions
    self.rewards[self.cursor_write_start:len(states)] = rewards
    self.states_next[self.cursor_write_start:len(states)] = states_next

    self.cursor_read_end = min(self.buffer_size, self.cursor_read_end + len(states))

def mlp(inp, inp_dim, outp_dim, track_scope=None, hidden=None, f=tf.tanh, bias_output=False):
    if not hidden:
        hidden = []

    layer_dims = [inp_dim] + hidden + [outp_dim]
    x = inp

    for i, (src_dim, tgt_dim) in enumerate(zip(layer_dims, layer_dims[1:])):
        Wi_name, bi_name = "W" +str(i), "b"+str(i)

        #Wi = ((track_scope and match_variable(Wi_name, track_scope))
        #      or tf.compat.v1.get_variable("W%i" % i, (src_dim, tgt_dim)))
        #Wi = tf.Variable(tf.random.normal(shape=(src_dim, tgt_dim),stddev=0.2),name = Wi_name)
        Wi = tf.compat.v1.get_variable("W%i" % i, (src_dim, tgt_dim))
        x = tf.matmul(x, Wi)

        final_layer = i == len(layer_dims) - 2
        if not final_layer or bias_output:
            #  bi = ((track_scope and match_variable(bi_name, track_scope))
            #        or tf.compat.v1.get_variable("b%i" % i, (tgt_dim,),
            #                           initializer=tf.zeros_initializer))
            #bi = tf.Variable(np.zeros(shape =(tgt_dim,)).astype(np.float32),name = bi_name)
            bi = tf.compat.v1.get_variable("b%i" % i, (tgt_dim,), initializer=tf.zeros_initializer)
            x += bi

        if not final_layer:
            x = f(x)

    return x

def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()
