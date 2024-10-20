#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn import functional
from torchnssd import Mamba2


class MambaActor(nn.Module):
    def __init__(
        self, cin, cout, d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size,
    ):
        super(MambaActor, self).__init__()
        self.input_linear = nn.Linear(cin, d_model, bias=False)
        self.mamba2 = Mamba2(d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, )
        # self.mlp_1 = nn.Linear(d_model, 128, bias=False)
        # self.mlp_2 = nn.Linear(128, 64, bias=False)
        # self.mlp_3 = nn.Linear(64, cout, bias=False)
        self.output_linear = nn.Linear(d_model, cout, bias = False)
        self.norm1 = nn.LayerNorm(d_model)
        self.activation = nn.ELU()
        
        self.chunk_size = chunk_size

    def forward(self, observations):
        '''
        observations [batch_size, n_state_buffer, cin]
        y [batch_size, cout]
        '''
        l = observations.shape[1]
        observations = functional.pad(observations, (0, 0, 0, (self.chunk_size - observations.shape[1] % self.chunk_size) % self.chunk_size))
        # forward pass
        y = self.input_linear(observations)    # y [batch_size, n_state_buffer, d_model]
        y = self.norm1(y)
        residual = y[:, -1, :]  # residual [batch_size, 1, d_model]
        y = self.mamba2(y)     # y [batch_size, n_state_buffer, d_model]
        y = y[:, -1, :].squeeze(1)     # y [batch_size, d_model]
        y = self.activation(y)
        y = self.output_linear(y)
        # y = self.norm1(y)
        # y = self.activation(y + residual)   # y [batch_size, d_model]
        # y = self.mlp_1(y)           # y[batch_size, 256]
        # y = self.activation(y)      # y[batch_size, 256]
        # y = self.mlp_2(y)           # y[batch_size, 128]
        # y = self.activation(y)      # y[batch_size, 128]
        # y = self.mlp_3(y)           # y[batch_size, cout]
        
        return y


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        buffer_size = 1,
        backbone="MAMBA",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        self.actor_obs_size = num_actor_obs
        self.buffer_size = buffer_size
        # Policy
        self.backbone = backbone
        if backbone == "MAMBA":
            self.actor = MambaActor(
                cin=input_dim_a,
                cout=num_actions,
                d_model=32,
                n_layer=4,
                d_state=32,
                d_conv=3,
                expand=2,
                headdim=16,
                chunk_size=8,
            )
        elif backbone == "MLP":
            self.actor = get_mlp_actor(input_dim_a, actor_hidden_dims, num_actions, activation)
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor {backbone}: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        obs_reformed = observations.view(-1, self.buffer_size, self.actor_obs_size)
        # forward pass, get the mean
        mean = self.actor(obs_reformed)  # mean: [batch_size, cout]

        # update the distribution
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        obs_reformed = observations.view(-1, self.buffer_size, self.actor_obs_size)
        # forward pass, get the mean
        mean = self.actor(obs_reformed)  # mean: [batch_size, cout]
        
        return mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
def get_mlp_actor(input_dim, actor_hidden_dims, num_actions, activation):
    actor_layers = []
    actor_layers.append(nn.Linear(input_dim, actor_hidden_dims[0]))
    actor_layers.append(activation)
    for layer_index in range(len(actor_hidden_dims)):
        if layer_index == len(actor_hidden_dims) - 1:
            actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
        else:
            actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
            actor_layers.append(activation)
    return nn.Sequential(*actor_layers)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
