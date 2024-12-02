#!/usr/bin/env python3

"""Actor critic."""

import numpy as np
import os
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

from mvp.backbones import vit

import math


ENCODER_REGULAR_VTH = 0.999
NEURON_VTH = 0.5
NEURON_CDECAY = 1 / 2
NEURON_VDECAY = 3 / 4
SPIKE_PSEUDO_GRAD_WINDOW = 0.5

class PseudoEncoderSpikeRegular(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Regular Spike for encoder """
    @staticmethod
    def forward(ctx, input):
        return input.gt(ENCODER_REGULAR_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

@torch.jit.export
class PopSpikeEncoderRegularSpike(nn.Module):
    """ Learnable Population Coding Spike Encoder with Regular Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std, device):
        """
        :param obs_dim: observation dimension
        :param pop_dim: population dimension
        :param spike_ts: spike timesteps
        :param mean_range: mean range
        :param std: std
        :param device: device
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoEncoderSpikeRegular.apply

        # print("DEBUG!!!obs_dim", obs_dim, pop_dim)
        # Compute evenly distributed mean and variance

        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)

        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std
        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs, batch_size):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: pop_spikes
        """
        obs = obs.view(-1, self.obs_dim, 1)
        # print("DEBUG1", torch.max(obs), " ", torch.min(obs))
        # Receptive Field of encoder population has Gaussian Shape

        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)  # NOTE if sim cant work, change rl to cpu
        # print("self.mean ", self.mean.shape, " ", self.mean)
        # print("pop_act ", pop_act.shape)
        # print("result", (obs - self.mean).shape, " ", (obs - self.mean)[0])
        # print("result2", pop_act[0])
        pop_volt = torch.zeros(batch_size, self.encoder_neuron_num, device=self.device)
        # print("pop_volt ", pop_volt.shape)
        pop_spikes = torch.zeros(batch_size, self.encoder_neuron_num, self.spike_ts, device=self.device)
        # print("pop_spikes ", pop_spikes.shape)
        # Generate Regular Spike Trains

        for step in range(self.spike_ts):
            pop_volt = pop_volt + pop_act
            pop_spikes[:, :, step] = self.pseudo_spike(pop_volt)
            pop_volt = pop_volt - pop_spikes[:, :, step] * ENCODER_REGULAR_VTH
            # print("pop_spikes ", pop_spikes.shape, " ", pop_spikes[:, :, step]) # NOTE (10, 170, 5)
        # print("DEBUG1", torch.sum(pop_spikes))
        return pop_spikes

# NOTE 2级
class PopSpikeDecoder(nn.Module):
    """ Population Coding Spike Decoder """
    # TODO 一定要仔细查看decoder的输出，也就是mean的范围
    def __init__(self, act_dim, pop_dim, output_activation=nn.ELU):  # Tanh  # nn.Identity
        """
        :param act_dim: action dimension
        :param pop_dim:  population dimension
        :param output_activation: activation function added on output
        """
        super().__init__()
        self.act_dim = act_dim
        self.pop_dim = pop_dim
        self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
        self.output_activation = output_activation()

    def forward(self, pop_act):
        """
        :param pop_act: output population activity
        :return: raw_act
        """
        pop_act = pop_act.view(-1, self.act_dim, self.pop_dim)
        # print("pop_act", pop_act.shape)
        # raw_act = self.output_activation(self.decoder(pop_act).view(-1, self.act_dim))
        raw_act = self.decoder(pop_act).view(-1, self.act_dim)
        # print("DEBUG1", torch.max(raw_act), " ", torch.min(raw_act))
        return raw_act

# NOTE 3级
# @torch.jit.export
class PseudoSpikeRect(torch.autograd.Function):
    """ Pseudo-gradient function for spike - Derivative of Rect Function """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()

class PreMLP(nn.Module):
    def __init__(self, obs_dim, device):
        super().__init__()
        self.obs_dim = obs_dim
        network_shape = [96, 192, 96]
        layer_num = len(network_shape)
        self.model = [nn.Linear(obs_dim, network_shape[0]),
                      nn.ELU()]
        if layer_num > 1:
            for layer in range(layer_num - 1):
                self.model.extend(
                    [nn.Linear(network_shape[layer], network_shape[layer + 1]),
                     nn.ELU()])
        self.model.extend([nn.Linear(network_shape[-1], obs_dim)])
        self.model = nn.Sequential(*self.model)

    def forward(self, state):
        out = self.model(state)
        return out

# NOTE 2级
class SpikeMLP(nn.Module):
    """ Spike MLP with Input and Output population neurons """
    def __init__(self, in_pop_dim, out_pop_dim, hidden_sizes, spike_ts, device):
        """
        :param in_pop_dim: input population dimension
        :param out_pop_dim: output population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        self.in_pop_dim = in_pop_dim
        self.out_pop_dim = out_pop_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_num = len(hidden_sizes)
        self.spike_ts = spike_ts
        self.device = device
        self.pseudo_spike = PseudoSpikeRect.apply
        # Define Layers (Hidden Layers + Output Population)
        # NOTE 240(obs_num*pop_num)->256->256->60(act_num*pop_num)
        self.hidden_layers = nn.ModuleList([nn.Linear(in_pop_dim, hidden_sizes[0])])
        if self.hidden_num > 1:
            for layer in range(1, self.hidden_num):
                self.hidden_layers.extend([nn.Linear(hidden_sizes[layer-1], hidden_sizes[layer])])
        self.out_pop_layer = nn.Linear(hidden_sizes[-1], out_pop_dim)

    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        LIF Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        # NOTE syn_func为一个nn.Linear
        # NOTE pre_layer_output为[env_num, obs_num*pop_num]
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)  # NOTE if sim cant work, change rl to cpu

        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def forward(self, in_pop_spikes, batch_size):
        """
        :param in_pop_spikes: input population spikes
        :param batch_size: batch size
        :return: out_pop_act
        """
        # Define LIF Neuron states: Current, Voltage, and Spike
        hidden_states = []
        # print("check device", self.device)
        for layer in range(self.hidden_num):
            hidden_states.append([torch.zeros(batch_size, self.hidden_sizes[layer], device=self.device)
                                  for _ in range(3)])
        # print("DEBUGHERE", len(hidden_states[0]), " ", hidden_states[0])
        out_pop_states = [torch.zeros(batch_size, self.out_pop_dim, device=self.device)
                          for _ in range(3)]
        out_pop_act = torch.zeros(batch_size, self.out_pop_dim, device=self.device)
        # Start Spike Timestep Iteration
        for step in range(self.spike_ts):
            in_pop_spike_t = in_pop_spikes[:, :, step]
            hidden_states[0][0], hidden_states[0][1], hidden_states[0][2] = self.neuron_model(
                self.hidden_layers[0], in_pop_spike_t,
                hidden_states[0][0], hidden_states[0][1], hidden_states[0][2]
            )
            if self.hidden_num > 1:
                for layer in range(1, self.hidden_num):
                    hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2] = self.neuron_model(
                        self.hidden_layers[layer], hidden_states[layer-1][2],
                        hidden_states[layer][0], hidden_states[layer][1], hidden_states[layer][2]
                    )
            out_pop_states[0], out_pop_states[1], out_pop_states[2] = self.neuron_model(
                self.out_pop_layer, hidden_states[-1][2],
                out_pop_states[0], out_pop_states[1], out_pop_states[2]
            )
            out_pop_act += out_pop_states[2]
            # print("DEBUGmu", out_pop_states[2])
        out_pop_act = out_pop_act / self.spike_ts
        # print("here", out_pop_act[2])
        return out_pop_act

# NOTE 1级（构建整个popsan）
class PopSpikeActor(nn.Module):
    """ Squashed Gaussian Stochastic Population Coding Spike Actor with Fix Encoder """

    def __init__(self, obs_dim, act_dim, en_pop_dim, de_pop_dim, hidden_sizes,
                 mean_range, std, spike_ts, device):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param en_pop_dim: encoder population dimension
        :param de_pop_dim: decoder population dimension
        :param hidden_sizes: list of hidden layer sizes
        :param mean_range: mean range for encoder
        :param std: std for encoder
        :param spike_ts: spike timesteps
        :param device: device
        """
        super().__init__()
        # self.act_dim = act_dim
        # self.premlp = PreMLP(obs_dim, device)
        self.encoder = PopSpikeEncoderRegularSpike(obs_dim, en_pop_dim, spike_ts, mean_range, std, device)
        self.snn = SpikeMLP(obs_dim*en_pop_dim, act_dim*de_pop_dim, hidden_sizes, spike_ts, device)
        self.decoder = PopSpikeDecoder(act_dim, de_pop_dim)
        log_std = -0.001 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # self.log_std = nn.Parameter(-0.5 * torch.zeros(act_dim))

    def forward(self, obs):
        """
        :param obs: observation
        :param batch_size: batch size
        :return: action scale with action limit
        """
        batch_size = obs.shape[0]
        in_pop_spikes = self.encoder(obs, batch_size)
        out_pop_activity = self.snn(in_pop_spikes, batch_size)
        mu = self.decoder(out_pop_activity)
        std = torch.exp(self.log_std)
        # out = self.decoder(out_pop_activity)
        # out = F.softplus(out)
        # alpha, beta = out[:, :self.act_dim], out[:, self.act_dim:]

        # print("DEBUGmu", torch.max(mu), " ", torch.min(mu))
        # print("DEBUGstd", std)
        #print("DEBUGmu", mu.shape, " ", obs.shape, " ", out.shape)

        return mu, std








###############################################################################
# States
###############################################################################

class ActorCritic(nn.Module):

    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        encoder_cfg,
        policy_cfg,
        net_type
    ):
        super(ActorCritic, self).__init__()
        assert encoder_cfg is None

        actor_hidden_dim = policy_cfg['pi_hid_sizes']
        critic_hidden_dim = policy_cfg['vf_hid_sizes']
        activation = nn.SELU()
        
        self.net_type = net_type
        # Policy
        if self.net_type == "ann":
        # ANN
            actor_layers = []
            actor_layers.append(nn.Linear(*obs_shape, actor_hidden_dim[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dim)):
                if l == len(actor_hidden_dim) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                    actor_layers.append(activation)
            self.actor = nn.Sequential(*actor_layers)

            print("check obs_shape!!!", obs_shape, *obs_shape)
            print("check actions_shape!!!", actions_shape, *actions_shape)
        elif self.net_type == "snn":
            # SNN
            self.actor = PopSpikeActor(*obs_shape, *actions_shape, 10, 10, actor_hidden_dim, (-5, 5), math.sqrt(0.15),
                                        spike_ts=5, device="cuda")

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)

        if self.net_type == "ann":
            # ANN
            self.init_weights(self.actor, actor_weights)
        
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    @torch.no_grad()
    def act(self, observations, states):
        if self.net_type == "ann":
            # ANN
            actions_mean = self.actor(observations)
        elif self.net_type == "snn":
            # SNN
            actions_mean, _ = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        # print("actions_mean", actions_mean.shape, actions_mean)
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        # print("",)
        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(observations)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
            None,  # dummy placeholder
        )

    @torch.no_grad()
    def act_inference(self, observations, states=None):
        if self.net_type == "ann":
            # ANN
            actions_mean = self.actor(observations)
        elif self.net_type == "snn":
            # SNN
            actions_mean, _ = self.actor(observations)
            
        return actions_mean

    def forward(self, observations, states, actions):
        if self.net_type == "ann":
            # ANN
            actions_mean = self.actor(observations)
        elif self.net_type == "snn":
            # SNN
            actions_mean, _ = self.actor(observations)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


###############################################################################
# Pixels
###############################################################################

_MODELS = {
    "vits-mae-hoi": "mae_pretrain_hoi_vit_small.pth",
    "vits-mae-in": "mae_pretrain_imagenet_vit_small.pth",
    "vits-sup-in": "sup_pretrain_imagenet_vit_small.pth",
    "vitb-mae-egosoup": "mae_pretrain_egosoup_vit_base.pth",
    "vitl-256-mae-egosoup": "mae_pretrain_egosoup_vit_large_256.pth",
}
_MODEL_FUNCS = {
    "vits": vit.vit_s16,
    "vitb": vit.vit_b16,
    "vitl": vit.vit_l16,
}


class Encoder(nn.Module):

    def __init__(self, model_name, pretrain_dir, freeze, emb_dim):
        super(Encoder, self).__init__()
        assert model_name in _MODELS, f"Unknown model name {model_name}"
        model_func = _MODEL_FUNCS[model_name.split("-")[0]]
        img_size = 256 if "-256-" in model_name else 224
        pretrain_path = os.path.join(pretrain_dir, _MODELS[model_name])
        self.backbone, gap_dim = model_func(pretrain_path, img_size=img_size)
        if freeze:
            self.backbone.freeze()
        self.freeze = freeze
        self.projector = nn.Linear(gap_dim, emb_dim)

    @torch.no_grad()
    def forward(self, x):
        feat = self.backbone.extract_feat(x)
        return self.projector(self.backbone.forward_norm(feat)), feat

    def forward_feat(self, feat):
        return self.projector(self.backbone.forward_norm(feat))


class PixelActorCritic(nn.Module):

    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        encoder_cfg,
        policy_cfg
    ):
        super(PixelActorCritic, self).__init__()
        assert encoder_cfg is not None

        # Encoder
        emb_dim = encoder_cfg["emb_dim"]

        self.obs_enc = Encoder(
            model_name=encoder_cfg["name"],
            pretrain_dir=encoder_cfg["pretrain_dir"],
            freeze=encoder_cfg["freeze"],
            emb_dim=emb_dim
        )
        self.state_enc = nn.Linear(states_shape[0], emb_dim)

        # AC params
        actor_hidden_dim = policy_cfg["pi_hid_sizes"]
        critic_hidden_dim = policy_cfg["vf_hid_sizes"]
        activation = nn.SELU()

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(emb_dim * 2, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for li in range(len(actor_hidden_dim)):
            if li == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[li], *actions_shape))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[li], actor_hidden_dim[li + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(emb_dim * 2, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for li in range(len(critic_hidden_dim)):
            if li == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[li], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dim[li], critic_hidden_dim[li + 1])
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(self.obs_enc)
        print(self.state_enc)
        print(self.actor)
        print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    @torch.no_grad()
    def act(self, observations, states):
        obs_emb, obs_feat = self.obs_enc(observations)
        state_emb = self.state_enc(states)
        joint_emb = torch.cat([obs_emb, state_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(joint_emb)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
            obs_feat.detach(),  # return obs features
        )

    @torch.no_grad()
    def act_inference(self, observations, states):
        obs_emb, _ = self.obs_enc(observations)
        state_emb = self.state_enc(states)
        joint_emb = torch.cat([obs_emb, state_emb], dim=1)
        actions_mean = self.actor(joint_emb)
        return actions_mean

    def forward(self, obs_features, states, actions):
        obs_emb = self.obs_enc.forward_feat(obs_features)
        state_emb = self.state_enc(states)
        joint_emb = torch.cat([obs_emb, state_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(joint_emb)

        return (
            actions_log_prob,
            entropy,
            value,
            actions_mean,
            self.log_std.repeat(actions_mean.shape[0], 1),
        )
