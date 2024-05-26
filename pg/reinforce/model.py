import torch
import numpy as np
import gym
from collections import namedtuple
import argparse

from pg.common import PrintWriter, CSVwriter
from typing import List


class Reinforce(torch.nn.Module):
    """ Simple Policy Gradient algorithm that does not rely on value learning.

    Args:
        policynet (torch.nn.Module): Pytorch policy module.
        log_dir (str, optional): Logging directory for the csv file. Defaults to "log".
    """
    Transition = namedtuple("Transition", "logprob reward")

    def __init__(self, policynet: torch.nn.Module, log_dir: str = "log"):
        super().__init__()
        self.policynet = policynet
        self.writers = [PrintWriter(flush=True), CSVwriter(log_dir)]

    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """ Return pi(a|s) distribution

        Args:
            state (torch.Tensor): State tensor

        Returns:
            torch.distributions.Categorical: Categorical distribution of actions given state (output of `policynet`)
        """
        return self.policynet(state)

    def accumulate_gradient(self, rollout: List[Transition], gamma: float) -> None:
        """ Calculate the gradients w.r.t policy parameters by performing backprob over the entire rollout.
        Call .backward() method. This will calculate and store the gradients within parameter tensors.
        Args:
            rollout (List[Transition]): List of Transitions
            gamma (float): discount factor
        """
        
        for idx, transition in enumerate(rollout):
            log_prob, reward = transition[0], transition[1]
            g= (gamma ** idx) * reward
            tmp = idx+1
            while tmp < len(rollout):
                next_reward = rollout[tmp][1]
                g += (gamma ** tmp) * next_reward
                tmp+=1
            loss = - (log_prob * g)
            loss.backward()

    def learn(self, args: argparse.Namespace, opt: torch.optim.Optimizer, env: gym.Env) -> None:
        """ Train the policy for <args.n_episodes> many episodes. First sample an 
        episode and then calculate the gradients using <accumulate_gradient> method.
        Log the training process to the writers.

        Args:
            args (argparse.Namespace): Hyperparameters
            opt (torch.optim.Optimizer): Optimizer for the policy network
            env (gym.Env): Environment that is used for sampling episodes
        """
        episodic_rewards = []

        for ix in range(args.n_episodes):
            episode_reward = 0
            state = env.reset()
            rollout = []
            for index in range(args.max_episode_len):
                state=torch.tensor(state)
                dist = self.policynet(state)
                action = (dist.sample()).item()
                next_state, reward, terminated,_ = env.step(action)
                log_prob_action = dist.log_prob(torch.tensor(action))
                transition = self.Transition(log_prob_action, reward) 
                rollout.append(transition)
                state = next_state
                
                # Don't forget to accumulate reward to episode_reward (logging purpose)
                episode_reward += reward

                if terminated:
                    break
            
            
            opt.zero_grad()
            self.accumulate_gradient(rollout, args.gamma)
            opt.step()

            episodic_rewards.append(episode_reward)
            
            if (ix + 1) % args.write_period == 0:
                for writer in self.writers:
                    writer(
                        dict(
                            Episode=ix+1,
                            Reward=np.mean(episodic_rewards[-args.log_window_length:])
                        )
                    )
            