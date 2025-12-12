import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch.nn.functional as F

class SNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_size=128, horizon=1):
        # features_dim = hidden_size
        super().__init__(observation_space, features_dim=hidden_size)

        self.last_metrics = None
        input_size = observation_space.shape[0]  # [obs, morph, phase]
        self.snn = ALIFCell(input_size, hidden_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, D]
        B = obs.shape[0]
        v, b = self.snn.init_state(B, device=obs.device)
        T = 10
        spk_seq = []
        v_seq = []
        b_seq = []

        for _ in range(T):
            s, (v, b) = self.snn(obs, (v, b))
            spk_seq.append(s)
            v_seq.append(v)
            b_seq.append(b)

        spk_seq = torch.stack(spk_seq, dim=0)  # [T, B, H]
        v_seq = torch.stack(v_seq)
        b_seq = torch.stack(b_seq)
        spk_mean = spk_seq.mean(dim=0)        # [B, H]

        self.last_metrics = {
            "spike_rate": spk_seq.float().mean().item(),
            "voltage_mean": v_seq.mean().item(),
            "voltage_var": v_seq.var().item(),
            "adapt_mean": b_seq.mean().item(),
            "adapt_var": b_seq.var().item(),
            "entropy": compute_entropy(spk_seq),
        }
        return spk_mean


def compute_entropy(spk_seq, eps=1e-8):
    p = spk_seq.float().mean(dim=(0, 1))  # [H]

    # Binary entropy function: -p log p - (1-p) log(1-p)
    entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))

    return entropy.mean().item()


"""
Simple adaptive LIF cell:
    v(t+1) = v(t) + ( -v(t) + W x(t) ) / tau_mem
    b(t+1) = b(t) + ( -b(t)/tau_adapt + s(t) )
    threshold(t) = v_th + adapt_scale * b(t)
"""
class ALIFCell(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        tau_mem=10.0,
        tau_adapt=100.0,
        v_th=1.0,
        adapt_scale=1.0,
    ):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size, bias=False)
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt
        self.v_th = v_th
        self.adapt_scale = adapt_scale

        # snnTorch surrogate spike function
        self.spike_fn = surrogate.fast_sigmoid()

    def init_state(self, batch_size, device=None):
        h = self.fc.out_features
        v = torch.zeros(batch_size, h, device=device)
        b = torch.zeros(batch_size, h, device=device)  # adaptation trace
        return v, b

    # x_t =  [batch, input_size] at time t
    def forward(self, x_t, state):
        v, b = state
        i_t = self.fc(x_t)
        v = v + ( -v + i_t ) / self.tau_mem # leaky integration

        thr = self.v_th + self.adapt_scale * b  # adaptive threshold
        s = self.spike_fn(v - thr)

        v = v - s * self.v_th
        b = b + ( -b / self.tau_adapt + s )  # update adaptation

        return s, (v, b)



class PoseALIFNet(nn.Module):
    def __init__(self, input_size, hidden_size, pose_dim, horizon):
        super().__init__()
        self.alif = ALIFCell(input_size, hidden_size,
                             tau_mem=10.0, tau_adapt=100.0,
                             v_th=1.0, adapt_scale=1.5)
        self.readout = nn.Linear(hidden_size, pose_dim * horizon)
        self.horizon = horizon
        self.pose_dim = pose_dim


    def forward(self, x):
        B, D = x.shape
        T = 10  # small unroll depth

        # repeat input T times: [T, B, D]
        x_seq = x.unsqueeze(0).repeat(T, 1, 1)

        state = self.alif.init_state(B, device=x.device)
        spk_seq = []
        for t in range(T):
            s, state = self.alif(x_seq[t], state)
            spk_seq.append(s)

        spk_seq = torch.stack(spk_seq, dim=0)    # [T, B, H]
        spk_mean = spk_seq.mean(dim=0)           # [B, H]
        out = self.readout(spk_mean)             # [B, horizon * pose_dim]
        return out



