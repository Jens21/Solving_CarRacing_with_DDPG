import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

from actor import Actor
from critic import Critic

class Network():
    train_step = 0
    losses_actor = []
    losses_critic = []

    def __init__(self, n_update_interval, gamma, n_heuristic_usage):
        self.n_update_interval = n_update_interval
        self.gamma = gamma
        self.n_heuristic_usage = n_heuristic_usage

        self.actor = Actor().to(device)
        self.critic = Critic().to(device)

        self.optimizer_actor = th.optim.Adam(self.actor.actor_policy.parameters(), lr=1e-4)
        self.optimizer_critic = th.optim.Adam(self.critic.critic_policy.parameters(), lr=1e-4)

    def create_input(self, s, speed):
        inp = th.from_numpy(s).flatten()
        inp = th.concat([inp, th.FloatTensor([speed])])

        return inp.float()

    def add_sample_to_buffer(self, s, speed, next_s, next_speed, r, action, replay_buffer):
        inp = self.create_input(s, speed)
        inp2 = self.create_input(next_s, next_speed)
        replay_buffer.add_sample([inp, inp2, r, action])

    def get_network_action(self, s, speed):
        inp = self.create_input(s, speed)
        inp = inp.to(device)[None]
        out = self.actor(inp)[0]

        return out.cpu().detach().numpy()

    def get_random_action(self):
        action = np.random.uniform(0, 1, 3)
        action[0] = 2 * action[0] - 1

        return action

    def get_heuristic_action(self, s, speed):
        action = np.zeros(3)

        left_part = s[22:32, 15:24].sum()
        right_part = s[22:32, 24:33].sum()
        diff = left_part - right_part
        action[0] = 0.1 * np.exp(np.abs(diff) / 50) * np.sign(diff)
        action[0] = np.clip(action[0], -1, 1)

        if speed < 50 and np.abs(action[0]) < 0.07 or speed < 20:
            action[1] = 1
        elif speed > 50:
            action[2] = 1

        return action

    def get_action(self, s, speed, itt):
        if itt < self.n_heuristic_usage/2: # get an action from the heuristic
            return self.get_heuristic_action(s, speed)
        elif itt < self.n_heuristic_usage: # get either an action from the heuristic or a random one
            if np.random.uniform(0, 1, 1) < 0.5:
                return self.get_heuristic_action(s, speed)
            else:
                return self.get_random_action()
        elif np.random.uniform(0, 1, 1) < 0.1:# get a random action
            return self.get_random_action()
        else:                                   # get an action from the network
            return self.get_network_action(s, speed)

    def create_critic_inputs(self, samples, out_actor):
        inp = th.concat([samples, out_actor], dim=1)

        return inp.float()

    def train(self, replay_buffer, batch_size):
        samples = replay_buffer.get_samples(batch_size)
        inputs = np.stack([samp[0] for samp in samples], axis=0)
        inputs = th.from_numpy(inputs).to(device).float()
        inputs_next = np.stack([samp[1] for samp in samples], axis=0)
        inputs_next = th.from_numpy(inputs_next).to(device).float()
        rewards = np.stack([samp[2] for samp in samples], axis=0)
        rewards = th.from_numpy(rewards).to(device).float()
        actions = np.stack([samp[3] for samp in samples], axis=0)
        actions = th.from_numpy(actions).to(device).float()

        # train the critic
        critic_inp = self.create_critic_inputs(inputs, actions)
        pred = self.critic(critic_inp).flatten(0)
        actor_out = self.actor.actor_target(inputs_next)
        critic_inp = self.create_critic_inputs(inputs_next, actor_out)
        critic_out = self.critic.critic_target(critic_inp).flatten(0)
        trg = rewards + self.gamma * critic_out

        self.optimizer_critic.zero_grad()
        loss_critic = F.mse_loss(pred, trg)
        # loss_critic = loss_critic if loss_critic<=1 else loss_critic/loss_critic.abs()
        loss_critic.backward()
        th.nn.utils.clip_grad_norm_(self.critic.critic_policy.parameters(), max_norm=0.1)
        self.optimizer_critic.step()
        self.losses_critic.append(loss_critic.item())


        # train the actor
        out_actor = self.actor(inputs)
        inp_critic = self.create_critic_inputs(inputs, out_actor)
        out_critic = self.critic(inp_critic)
        loss_actor = -out_critic.mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        th.nn.utils.clip_grad_norm_(self.actor.actor_policy.parameters(), max_norm=0.1)
        self.optimizer_actor.step()
        self.losses_actor.append(loss_actor.item())

        self.train_step += 1
        if self.train_step % self.n_update_interval == 0:
            self.actor.update_networks()
            self.critic.update_networks()

    def plot_losses(self, path):
        plt.figure(figsize=(8,8))
        plt.plot(np.arange(len(self.losses_actor[100:])), self.losses_actor[100:])
        plt.savefig(os.path.join(path,'actor_losses.png'))
        plt.close()

        plt.figure(figsize=(8,8))
        plt.plot(np.arange(len(self.losses_critic[100:])), self.losses_critic[100:])
        plt.savefig(os.path.join(path,'critic_losses.png'))
        plt.close()