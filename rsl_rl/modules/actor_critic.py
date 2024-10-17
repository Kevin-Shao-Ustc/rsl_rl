#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from torchnssd import BiMamba2


class MambaActor(nn.Module):
    def __init__(
        self, cin, cout, d_model, n_layer, d_state, d_conv, expand, headdim, chunk_size, n_state_buffer,
    ):
        super(MambaActor, self).__init__()
        self.bimamba2 = BiMamba2(
            cin=cin, cout=cout, d_model=d_model, n_layer=n_layer, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim, chunk_size=chunk_size
        )
        self.output_linear = nn.Linear(n_state_buffer, 1)
        self.n_state_buffer = n_state_buffer

    def forward(self, x, buffer):
        '''
        x [batch_size, cin]
        buffer [batch_size, cin, n_state_buffer]
        y [batch_size, cout]
        '''
        # add the new input to the buffer
        buffer = torch.cat([buffer, x.unsqueeze(2)], dim=2)
        # remove the oldest input from the buffer
        buffer = buffer[:, :, -self.n_state_buffer:]

        # forward pass
        y = self.bimamba2(buffer)  # y [batch_size, cout, n_state_buffer]
        
        y = self.output_linear(y).squeeze(2) # y [batch_size, cout]

        return y, buffer


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
        # Policy
        self.backbone = backbone
        if backbone == "MAMBA":
            self.actor = MambaActor(
                cin=input_dim_a,
                cout=num_actions,
                d_model=64,
                n_layer=4,
                d_state=32,
                d_conv=3,
                expand=2,
                headdim=32,
                chunk_size=32,
                n_state_buffer=8
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

        # Buffer for recurrent policies
        self.register_buffer('buffer', None)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        # Reset buffer for recurrent policies
        self.buffer = None

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
        if self.buffer is None or self.buffer.shape[0] != observations.shape[0]:
            # Create buffer for recurrent policies
            self.buffer = observations.unsqueeze(2).repeat(1, 1, self.actor.n_state_buffer)
        
        # forward pass, get the mean and the updated buffer
        mean, new_buffer = self.actor(observations, self.buffer)  # mean: [cout], new_buffer: [buffer_length, cin]

        # update the distribution
        self.distribution = Normal(mean, mean * 0.0 + self.std)
        # update the buffer, detach to avoid backpropagation
        self.buffer = new_buffer.detach()

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        if self.buffer is None or self.buffer.shape[0] != observations.shape[0]:
            # Create buffer for recurrent policies
            self.buffer = observations.unsqueeze(2).repeat(1, 1, self.actor.n_state_buffer)

        # forward pass, get the mean and the updated buffer
        actions_mean, new_buffer = self.actor(observations, self.buffer)
        # update the buffer, detach to avoid backpropagation
        self.buffer = new_buffer.detach()
        return actions_mean

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
