import sys
import math
import time
import copy
import random
from datetime import datetime
from collections import deque, namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal, Normal

import gym
from ENV.bipedalWalker import BipedalWalker

from utils import *

class NAF(nn.Module):
    def __init__(self, state_size, action_size,layer_size, seed):
        super(NAF, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size

        self.head_1 = nn.Linear(self.input_shape, layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)
        self.ff_1 = nn.Linear(layer_size, layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)
        self.action_values = nn.Linear(layer_size, action_size)
        self.value = nn.Linear(layer_size, 1)
        self.matrix_entries = nn.Linear(layer_size, int(self.action_size*(self.action_size+1)/2))



    def forward(self, input_, action=None):
        """

        """

        x = torch.relu(self.head_1(input_))
        x = self.bn1(x)
        x = torch.relu(self.ff_1(x))
        action_value = torch.tanh(self.action_values(x))
        entries = torch.tanh(self.matrix_entries(x))
        V = self.value(x)

        action_value = action_value.unsqueeze(-1)

        # create lower-triangular matrix
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size))

        # get lower triagular indices
        tril_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)

        # fill matrix with entries
        L[:, tril_indices[0], tril_indices[1]] = entries
        L.diagonal(dim1=1,dim2=2).exp_()

        # calculate state-dependent, positive-definite square matrix
        P = L*L.transpose(2, 1)

        Q = None
        if action is not None:

            # calculate Advantage:
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P), (action.unsqueeze(-1) - action_value))).squeeze(-1)

            Q = A + V


        # add noise to action mu:
        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        #dist = Normal(action_value.squeeze(-1), 1)
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)


        return action, Q, V


class DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 PER,
                 LR,
                 Nstep,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 NUPDATES,
                 seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            Network (str): dqn network type
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.nstep = Nstep
        self.UPDATE_EVERY = UPDATE_EVERY
        self.NUPDATES = NUPDATES
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.per = PER

        self.action_step = 4
        self.last_action = None

        # Q-Network
        self.qnetwork_local = NAF(state_size, action_size,layer_size, seed)
        self.qnetwork_target = NAF(state_size, action_size,layer_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)

        # Replay memory
        if PER == True:
            print("Using Prioritized Experience Replay")
            self.memory = PrioritizedReplay(BUFFER_SIZE, BATCH_SIZE, seed, n_step=self.nstep)
        else:
            print("Using Regular Experience Replay")
            self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed, self.GAMMA, self.nstep)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                Q_losses = []
                for _ in range(self.NUPDATES):
                    experiences = self.memory.sample()
                    if self.per == True:
                        loss = self.learn_per(experiences)
                    else:
                        loss = self.learn(experiences)
                    self.Q_updates += 1
                    Q_losses.append(loss)

                return np.mean(Q_losses)



    def act(self, state):
        """Calculating the action

        Params
        ======
            state (array_like): current state

        """

        state = torch.from_numpy(state).float()

        self.qnetwork_local.eval()
        with torch.no_grad():
            action, _, _ = self.qnetwork_local(state.unsqueeze(0))
            #action = action.cpu().squeeze().numpy() + self.noise.sample()
            #action = np.clip(action, -1,1)[0]
        self.qnetwork_local.train()
        return action.cpu().squeeze().numpy().reshape((self.action_size,))



    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, V_ = self.qnetwork_target(next_states)

        # Compute Q targets for current states
        V_targets = rewards + (self.GAMMA**self.nstep * V_ * (1 - dones))

        # Get expected Q values from local model
        _, Q, _ = self.qnetwork_local(states, actions)

        # Compute loss
        loss = F.mse_loss(Q, V_targets)

        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)



        return loss.detach().cpu().numpy()

    def learn_per(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones, idx, weights = experiences

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(np.float32(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, V_ = self.qnetwork_target(next_states)

        # Compute Q targets for current states
        V_targets = rewards + (self.GAMMA**self.nstep * V_ * (1 - dones))

        # Get expected Q values from local model
        _, Q, _ = self.qnetwork_local(states, actions)

        # Compute loss
        td_error = V_targets - Q
        loss = (td_error.pow(2)*weights).mean()
        # Minimize the loss
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        # update per priorities
        self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))


        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)


def run(frames=100000):
    max_episodes = 10000
    max_steps = 500
    score_list = []
    best_score = -1000000

    tic = time.time()

    for i in range(max_episodes):

        state = env.reset()
        score = 0
        losses =[]

        for j in range(max_steps):

            if '--visu' in sys.argv:
                env.render()

            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            loss = agent.step(state, action, reward, next_state, done)
            if loss:
                losses.append(loss)

            state = next_state
            score += reward

            tac = time.time()
            print("\033[0;1;4;97m", end='')
            print("Episode:", end = "")
            print("\033[0;97m", end='')
            print(" {}    ".format(i),end='')
            print("\033[3;4;91m", end='')
            print("temps total : {} secondes\r".format(int(tac - tic)), end='')

            if done:

                break

        writer.add_scalar(name + '/reward', score, i)
        writer.add_scalar(name + '/loss', np.mean(losses), i)
        writer.flush()

        score_list.append(score)

        if i % 10 == 0:
            print("\033[0;1;4;97m", end='')
            print("Episode:", end = "")
            print("\033[0;97m", end='')
            print(" {}                                             ".format(i))
            print("total reward: {:.5}  avg reward (last 10): {:.5}".format(score,np.mean(score_list[max(0, i-10):(i+1)])))
            if "--save" in sys.argv:
                arg_index = sys.argv.index("--save")
                save_name = sys.argv[arg_index + 1]
                torch.save(agent.qnetwork_local.state_dict(), "savedir/"+save_name+"/local.pth")
                torch.save(agent.qnetwork_target.state_dict(), "savedir/"+save_name+"/target.pth")
        if ("--savemax" in sys.argv) and (score > best_score):
            best_score = score
            arg_index = sys.argv.index("--save")
            save_name = sys.argv[arg_index + 1]
            torch.save(agent.qnetwork_local.state_dict(), "savedir/"+save_name+"/local.pth")
            torch.save(agent.qnetwork_target.state_dict(), "savedir/"+save_name+"/target.pth")

    return np.mean(score_list[-100:])



if __name__ == "__main__":

    frames = 40000
    seed = np.random.randint(100000)
    per = False
    BUFFER_SIZE = 100000
    BATCH_SIZE = 128
    LAYER_SIZE = 1000
    nstep = 1
    GAMMA = 0.99
    TAU = 0.01
    LR = 0.001
    UPDATE_EVERY = 1
    NUPDATES = 3
    name = "walker"

    paramDict ={"name":name,
                "seed":seed,
                "BufferSize":BUFFER_SIZE,
                "per":per,
                "batch_size":BATCH_SIZE,
                "layer_size":LAYER_SIZE,
                "nStep":nstep,
                "gamma":GAMMA,
                "tau":TAU,
                "learningRate":LR,
                "updateEvery":UPDATE_EVERY,
                "nUpdate":NUPDATES}

    np.random.seed(seed)
    env = BipedalWalker()

    now = datetime.now()
    writer = SummaryWriter('logdir/'+ now.strftime("%Y%m%d-%H%M%S") + "/")
    writer.add_hparams(paramDict,{})

    env.seed(seed)
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    agent = DQN_Agent(state_size=state_size,
                        action_size=action_size,
                        layer_size=LAYER_SIZE,
                        BATCH_SIZE=BATCH_SIZE,
                        BUFFER_SIZE=BUFFER_SIZE,
                        PER=per,
                        LR=LR,
                        Nstep=nstep,
                        TAU=TAU,
                        GAMMA=GAMMA,
                        UPDATE_EVERY=UPDATE_EVERY,
                        NUPDATES=NUPDATES,
                        seed=seed)

    #writer.add_graph(agent.qnetwork_local)
    #writer.close()

    if "--load" in sys.argv:
        print("loading weights")
        arg_index = sys.argv.index("--load")
        save_name = sys.argv[arg_index + 1]
        agent.qnetwork_local.load_state_dict(torch.load("savedir/" + save_name + "/local.pth"))
        agent.qnetwork_target.load_state_dict(torch.load("savedir/" + save_name + "/target.pth"))
        print("weights loaded")

    run(frames = frames)
