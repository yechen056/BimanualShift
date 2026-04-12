import torch
import torch.nn as nn


class ActionGenerator(nn.Module):
    """Lightweight residual injector that augments the existing SkillManager path."""

    def __init__(self, visual_dim=128, lang_dim=128, proprio_dim=8, hidden_dim=256):
        super().__init__()
        self.visual_dim = visual_dim
        self.lang_dim = lang_dim
        self.proprio_dim = proprio_dim
        self.net = nn.Sequential(
            nn.Linear(visual_dim * 2 + lang_dim + proprio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, lang_dim * 2),
        )

    def forward(self, mask_right, mask_left, proprio, lang_tokens):
        right_ctx = mask_right.mean(dim=1)
        left_ctx = mask_left.mean(dim=1)
        lang_ctx = lang_tokens.mean(dim=1)
        proprio_ctx = proprio.float()
        features = torch.cat([right_ctx, left_ctx, lang_ctx, proprio_ctx], dim=-1)
        residuals = self.net(features)
        right_residual, left_residual = residuals.chunk(2, dim=-1)
        right_residual = right_residual.unsqueeze(1).expand(-1, lang_tokens.shape[1], -1)
        left_residual = left_residual.unsqueeze(1).expand(-1, lang_tokens.shape[1], -1)
        return {
            "right_residual": right_residual,
            "left_residual": left_residual,
        }
