# diffusion.py

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F


class TemporalUNet1D(nn.Module):
    def __init__(self, obs_dim: int, cond_channels: int, base_channels: int = 128):
        super().__init__()
        in_ch = obs_dim + cond_channels

        self.down1 = nn.Conv1d(in_ch, base_channels, kernel_size=3, padding=1)
        self.down2 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.mid  = nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)

        self.up1  = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.up2  = nn.Conv1d(base_channels, obs_dim, kernel_size=3, padding=1)

    def forward(self, x_seq: torch.Tensor, cond_seq: torch.Tensor) -> torch.Tensor:
        # x_seq, cond_seq: (B, T, feat)
        B, T, _ = x_seq.shape
        B2, T2, _ = cond_seq.shape
        assert B == B2 and T == T2

        # concat along feature axis, then go to (B, C, T)
        h = torch.cat([x_seq, cond_seq], dim=-1)   # (B, T, in_ch)
        h = h.permute(0, 2, 1)                    # (B, in_ch, T)

        d1 = F.relu(self.down1(h))               # (B, C, T)
        d2 = F.relu(self.down2(d1))              # (B, 2C, T//2)

        m  = F.relu(self.mid(d2))                # (B, 2C, T//2)

        u1 = F.relu(self.up1(m))                 # (B, C, ~T)
        # match temporal length to d1
        if u1.size(-1) > d1.size(-1):
            u1 = u1[..., : d1.size(-1)]
        elif u1.size(-1) < d1.size(-1):
            pad = d1.size(-1) - u1.size(-1)
            u1 = F.pad(u1, (0, pad))

        out = self.up2(u1)                       # (B, obs_dim, T)
        out = out.permute(0, 2, 1)               # (B, T, obs_dim)
        return out


class PoseDiffusion(nn.Module):
    def __init__( self,
        D_target,
        D_cond,
        future_len = 200,
        n_steps = 100,
        hidden = 256,
        beta_start = 1e-4,
        beta_end = 2e-2 ):

        super().__init__()
        self.obs_dim = D_target
        self.cond_dim = D_cond
        self.future_len = future_len
        self.n_steps = n_steps

        # diffusion schedule
        betas = torch.linspace(beta_start, beta_end, n_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

        # time embedding
        self.time_dim = hidden
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        # temporal UNet; cond_channels = cond_dim + time_dim
        self.unet = TemporalUNet1D(
            obs_dim=self.obs_dim,
            cond_channels=self.cond_dim + self.time_dim,
            base_channels=hidden )

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        assert x0.shape == noise.shape

        alpha_bar_t = self.alpha_bars[t].view(B, 1, 1)  # (B,1,1)

        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:

        B, T, F = x_t.shape
        assert F == self.obs_dim

        # time embedding
        t_norm = t.float().view(B, 1) / float(self.n_steps)
        t_feat = self.t_embed(t_norm)  # (B, time_dim)

        # expand over time axis
        t_seq = t_feat.unsqueeze(1).expand(-1, T, -1)  # (B, T, time_dim)
        cond_sq = cond_vec.unsqueeze(1).expand(-1, T, -1)  # (B, T, D_cond)

        cond_full = torch.cat([cond_sq, t_seq], dim=-1)  # (B, T, D_cond + time_dim)

        eps = self.unet(x_t, cond_full)  # (B, T, obs_dim)
        return eps

    @torch.no_grad()
    def sample(self, cond_vec: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:

        if n_steps is None:
            n_steps = self.n_steps

        device = self.betas.device
        B = cond_vec.size(0)
        T = self.future_len

        x_t = torch.randn(B, T, self.obs_dim, device=device)

        for step in reversed(range(n_steps)):
            t = torch.full((B,), step, device=device, dtype=torch.long)

            eps = self.forward(x_t, t, cond_vec)

            beta_t = self.betas[t].view(B, 1, 1)
            alpha_t = self.alphas[t].view(B, 1, 1)
            alpha_bar_t = self.alpha_bars[t].view(B, 1, 1)

            if step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            x_t = (
                    1.0 / torch.sqrt(alpha_t)
                    * (x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps)
                    + torch.sqrt(beta_t) * noise        )

        return x_t


def train_diffusion(traj_file = "quadruped_morph_trajectories.npz",
    past_len= 10,  future_len = 20,
    batch_size = 128,
    num_epochs = 50,
    lr = 1e-4,
    n_steps= 100,
    hidden = 256,
    save_path: str = "quadruped_morph_diffusion_weights.pt",
    imitation_obs_indices=None,
    device: str | None = None ):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = QuadrupedMorphDiffusionDataset(
        traj_path=traj_file,
        past_len=past_len,
        future_len=future_len,
        imitation_obs_indices=imitation_obs_indices,
        device=device )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    diff_model = PoseDiffusion(
        D_target=dataset.D_target,  # = obs_dim
        D_cond=dataset.D_cond,  # = past_len*obs_dim + morph_dim
        future_len=future_len,
        n_steps=n_steps,
        hidden=hidden,
    ).to(device)

    optimizer = torch.optim.Adam(diff_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training
    loss_history = []

    print_model_info(diff_model) #debug

    for epoch in range(num_epochs):
        diff_model.train()
        epoch_loss = 0.0
        num_batches = 0

        for x0, cond in loader:
            x0 = x0.to(device)
            cond = cond.to(device)
            B = x0.size(0)

            # sample diffusion step t for each item in batch
            t = torch.randint(0, diff_model.n_steps, (B,), device=device)

            noise = torch.randn_like(x0)
            x_t = diff_model.q_sample(x0, t, noise)

            # predict the noise
            pred = diff_model(x_t, t, cond)

            loss = criterion(pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")
        total_norm = 0
        for p in diff_model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item()
        print("  grad_norm:", total_norm)

    # ----------- SAVE -------------
    torch.save(
        {
            "state_dict": diff_model.state_dict(),
            "config": {
                "traj_file": traj_file,
                "past_len": past_len,
                "future_len": future_len,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "lr": lr,
                "n_steps": n_steps,
                "hidden": hidden,
                "imitation_obs_indices": imitation_obs_indices
            },
        },
        save_path,
    )
    print("Saved diffusion weights to:", save_path)
    return diff_model, loss_history


class QuadrupedMorphDiffusionDataset(Dataset):
    def __init__(self,
            traj_path: str,
            past_len = 10,
            future_len = 10,
            imitation_obs_indices=None,
            device: str = "cpu",):

        super().__init__()

        data = np.load(traj_path)

        # expected shapes:
        # obs:   (N, obs_dim)
        # morph: (N, morph_dim)
        self.obs = data["obs"].astype(np.float32)
        self.morph = data["morph"].astype(np.float32)

        # dims
        self.obs_dim = self.obs.shape[1]
        self.morph_dim = self.morph.shape[1]

        self.past_len = past_len
        self.future_len = future_len
        self.device = device
        if imitation_obs_indices is None:
            self.imitation_obs_indices = None
            self.target_dim = self.obs_dim
        else:
            self.imitation_obs_indices = np.asarray(imitation_obs_indices, dtype=np.int64)
            self.target_dim = self.imitation_obs_indices.shape[0]
        self.T = len(self.obs)

        # number of valid sliding-window samples
        self.max_index = self.T - (past_len + future_len)

        print(f"[QuadrupedDataset] obs_dim={self.obs_dim}, morph_dim={self.morph_dim}")
        print(f"[QuadrupedDataset] target_dim={self.target_dim}")
        print(f"[QuadrupedDataset] total samples possible={self.max_index}")

        # D_target is per-time-step dimension (obs_dim)
        # D_cond   is the flattened conditioning vector
        self.D_target = self.target_dim
        self.D_cond = self.past_len * self.target_dim + self.morph_dim


    def __len__(self):
        return max(0, self.max_index)


    def __getitem__(self, idx: int):
        # slice past & future
        past_obs = self.obs[idx: idx + self.past_len]  # (P, obs_dim)
        future_obs = self.obs[ idx + self.past_len: idx + self.past_len + self.future_len ]

        if self.imitation_obs_indices is not None:
            past_obs = past_obs[:, self.imitation_obs_indices]
            future_obs = future_obs[:, self.imitation_obs_indices]

        morph_vec = self.morph[idx + self.past_len]
        past_flat = past_obs.reshape(-1)  # (P*obs_dim)
        cond = np.concatenate([past_flat, morph_vec], axis=0)

        # x0 is now a SEQUENCE: (F, obs_dim)
        x0 = torch.tensor(future_obs, device=self.device, dtype=torch.float32)
        cond = torch.tensor(cond, device=self.device, dtype=torch.float32)

        return x0, cond


class DiffusionPoseTemplate:
    def __init__(self, pose_cycle: np.ndarray):
        """
        pose_cycle: (cycle_steps, obs_dim) array
        """
        self.pose_cycle = np.asarray(pose_cycle, dtype=np.float32)
        self.cycle_steps = self.pose_cycle.shape[0]

    def __call__(self, morph_vec, phase: float) -> np.ndarray:
        # phase in [0,1); map to index
        idx = int(phase * self.cycle_steps) % self.cycle_steps
        return self.pose_cycle[idx]


def sample_full_cycle_from_morph( diff_model,
    meta,
    morph_vec,
    cycle_steps=200,
    device="cuda"):

    past_len = meta["past_len"]
    morph_dim = meta["morph_dim"]
    future_len = meta["future_len"]

    obs_dim   = diff_model.obs_dim
    cond_dim  = diff_model.cond_dim

    if cycle_steps is None:
        cycle_steps = future_len


    past_flat = torch.zeros(past_len * obs_dim, device=device)
    morph_vec = np.asarray(morph_vec, dtype=np.float32)
    assert morph_vec.shape[0] == morph_dim, (
        f"Got morph_vec length {morph_vec.shape[0]}, expected {morph_dim}"
    )
    morph_t = torch.tensor(morph_vec, device=device, dtype=torch.float32)

    cond_vec = torch.cat([past_flat, morph_t], dim=0).unsqueeze(0)  # (1, D_cond)

    # Sample a future sequence from diffusion
    with torch.no_grad():
        seq = diff_model.sample(cond_vec)  # (1, future_len, obs_dim)

    seq = seq[0].cpu().numpy()  # (future_len, obs_dim)

    # If needed, resample to 'cycle_steps'
    if seq.shape[0] != cycle_steps:
        idxs = np.linspace(0, seq.shape[0] - 1, cycle_steps).astype(int)
        seq = seq[idxs]

    return seq

def print_model_info(model: PoseDiffusion):
    print("\n=== Diffusion Model Summary ===")
    print("obs_dim      :", model.obs_dim)
    print("cond_dim     :", model.cond_dim)
    print("future_len   :", model.future_len)
    print("n_steps      :", model.n_steps)
    print("time_dim     :", model.time_dim)

    # diffusion schedule info
    print("beta range   :", (model.betas.min().item(), model.betas.max().item()))
    print("alpha range  :", (model.alphas.min().item(), model.alphas.max().item()))
    print("alpha_bar[0] :", model.alpha_bars[0].item())
    print("alpha_bar[-1]:", model.alpha_bars[-1].item())

    # parameter counts
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameters   :", f"total={total:,}  trainable={trainable:,}")

    # UNet structure
    unet = model.unet
    print("\n--- UNet Channels ---")
    print("down1:", unet.down1)
    print("down2:", unet.down2)
    print("mid  :", unet.mid)
    print("up1  :", unet.up1)
    print("up2  :", unet.up2)
    print("==========================\n")