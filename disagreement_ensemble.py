"""Bootstrapped three-head disagreement estimator over frozen RSSM posterior features.

Auxiliary diagnostic only: this module does not change the Dreamer world model or actor.
"""
import torch
from torch import nn
from torch.nn import functional as F


class DisagreementEnsemble(nn.Module):
    def __init__(self, feature_dim, action_dim, target_dim=128, hidden_dim=128,
                 num_heads=3, seed=2026):
        super().__init__()
        if num_heads < 2:
            raise ValueError("At least two heads are required")
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim + self.action_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.target_dim),
            )
            for _ in range(num_heads)
        ])
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(self.feature_dim, self.target_dim, generator=generator)
        projection /= self.feature_dim ** 0.5
        self.register_buffer("target_projection", projection)

    def project_target(self, next_feature):
        target = next_feature.detach().float() @ self.target_projection
        return F.layer_norm(target, (self.target_dim,))

    def forward(self, feature, action):
        x = torch.cat((feature.detach().float(), action.detach().float()), dim=-1)
        return torch.stack([head(x) for head in self.heads], dim=0)

    def disagreement(self, feature, action):
        predictions = self(feature, action)
        return predictions.float().var(dim=0, unbiased=False).mean(dim=-1)

    def bootstrap_loss(self, feature, action, next_feature, bootstrap_p=0.8):
        if not 0.0 < bootstrap_p <= 1.0:
            raise ValueError("bootstrap_p must be in (0, 1]")
        predictions = self(feature, action)
        target = self.project_target(next_feature)
        errors = (predictions.float() - target.unsqueeze(0)).square().mean(-1)
        mask = (torch.rand_like(errors) < bootstrap_p).float()
        empty_heads = mask.sum(dim=1) == 0
        if empty_heads.any():
            mask[empty_heads, 0] = 1.0
        per_head = (errors * mask).sum(dim=1) / mask.sum(dim=1)
        return per_head.mean(), per_head.detach()
