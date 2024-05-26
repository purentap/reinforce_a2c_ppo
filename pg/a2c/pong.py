from typing import Tuple
import argparse
import torch
import gym
import numpy as np
import torch.nn as nn
from pg.a2c.model import A2C
from pg.a2c.vecenv import ParallelEnv
from pg.common import ResizeAndScalePong, NoopResetEnv
from pg.common import DerivativeEnv, DoubleActionPong


class GruNet(torch.nn.Module):
    """ Gru network that combines value and policy functions.

    Args:
        in_size (int): Channel size of the state tensor
        out_size (int): Action size (number of categories)
        hidden (int, optional): Hidden neuron size. Defaults to 128.
    """

    def __init__(self, in_size: int, out_size: int, hidden: int = 128):
        super().__init__()

        #gues we dont need this bc of wrappers?
        #input shaped [1, 80, 80]
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size= 8, stride=4),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU()
                )
        
        self.linear_nn = nn.Sequential(
			nn.Linear(80*80, 512),
			#nn.ReLU(),
			
        )
        self.body = nn.GRUCell(input_size=512, hidden_size=hidden)
        self.policynet = nn.Linear(in_features= hidden, out_features= out_size)
        self.valuenet = nn.Linear(in_features=hidden, out_features= 1)

    def forward(self,
                state: torch.Tensor,
                gru_hx: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Return policy logits, value, and the gru hidden state for the given state

        Args:
            state (torch.Tensor): State tensor
            gru_hx (torch.Tensor): gru hidden state

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                policy logits, value, and gru hidden state
        """
        state = state.float()
        #x = self.feature_extractor(state)
        x = state
        x = x.view(x.shape[0], -1)
        x = self.linear_nn(x)
        
        gru_hx = self.body(x, gru_hx)
        value = self.valuenet(gru_hx)
        logits = self.policynet(gru_hx)

        return logits, value, gru_hx


def make_env() -> gym.Env:
    """ Environment creating function """
    env = gym.make("ALE/Pong-v5")
    env = ResizeAndScalePong(env)
    env = DerivativeEnv(env)
    env = NoopResetEnv(env, 20, 20)
    env = DoubleActionPong(env)
    return env


def main(args):
    """ Start the learning process with the given arguments """
    seed = args.seed or np.random.randint(2**10, 2**30)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vecenv = ParallelEnv(args.nenv, make_env, seed=seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # We need to initialize an environment to get dimensions
    env = make_env()
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.n

    network = GruNet(in_size, out_size, args.hidden_size)
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    agent = A2C(network, args, vecenv, optimizer)
    agent.to(args.device)
    # We no longer need to keep this environment
    del env

    agent.learn()
    agent.save("models/pong.b")

    vecenv.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A2C with Pong")
    parser.add_argument("--nenv", type=int,
                        help="Number of environemnts run in parallel",
                        default=16)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed of the experiment")
    parser.add_argument("--lr", type=float, help="Learning rate",
                        default=3e-4)
    parser.add_argument("--device", type=str, help="Torch device",
                        default="cpu")
    parser.add_argument("--timesteps", type=int,
                        help="Number of timesteps to run learning",
                        default=int(1e7))
    parser.add_argument("--n-step", type=int,
                        help="Length of the rollout",
                        default=10)
    parser.add_argument("--hidden-size", type=int,
                        help="Number of neurons in the hidden layers and gru",
                        default=256)
    parser.add_argument("--gamma", type=float,
                        help="Discount factor",
                        default=0.995)
    parser.add_argument("--gae_lambda", type=float,
                        help="lambda coefficient",
                        default=0.95)
    parser.add_argument("--entropy_coef", type=float,
                        help="Entropy coefficient",
                        default=0.01)
    parser.add_argument("--value_coef", type=float,
                        help="Value coefficient",
                        default=0.5)
    parser.add_argument("--write-period", type=int,
                        help="Logging period (interms of timesteps)",
                        default=16 * 10 * 100)
    parser.add_argument("--log-window-length", type=int,
                        help="Last n episodic rewards to log",
                        default=25)
    parser.add_argument("--log-dir", type=str,
                        help="Logging directory",
                        default=None)
    args = parser.parse_args()

    main(args)
