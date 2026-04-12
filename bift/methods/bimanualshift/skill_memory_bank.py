from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MemoryEntry:
    key: torch.Tensor
    action: torch.Tensor
    score: float
    age: int


class SkillMemoryBank:
    """Lightweight retrieval memory for BimanualShift-style inference."""

    def __init__(self, capacity: int = 256, sim_threshold: float = 0.65, blend_alpha: float = 0.8):
        self.capacity = capacity
        self.sim_threshold = sim_threshold
        self.blend_alpha = blend_alpha
        self.entries: List[MemoryEntry] = []
        self._age = 0

    @staticmethod
    def build_key(lang_goal_emb: torch.Tensor, voxel_grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lang_goal_emb.dim() > 1:
            lang_key = lang_goal_emb.mean(dim=0)
        else:
            lang_key = lang_goal_emb
        if voxel_grid is None:
            key = lang_key
        else:
            vis_stats = voxel_grid.float().mean(dim=tuple(range(1, voxel_grid.dim())))
            vis_stats = vis_stats.reshape(-1)
            key = torch.cat([lang_key.reshape(-1), vis_stats], dim=0)
        return F.normalize(key.float(), dim=0)

    def retrieve(self, key: torch.Tensor) -> Tuple[Optional[torch.Tensor], float]:
        if not self.entries:
            return None, -1.0
        key = F.normalize(key.float(), dim=0)
        sims = []
        for e in self.entries:
            sim = F.cosine_similarity(key.unsqueeze(0), e.key.unsqueeze(0), dim=1).item()
            sims.append(sim)
        best_idx = int(torch.tensor(sims).argmax().item())
        best_score = sims[best_idx]
        if best_score < self.sim_threshold:
            return None, best_score
        return self.entries[best_idx].action.clone(), best_score

    def update(self, key: torch.Tensor, action: torch.Tensor, success: bool = True):
        if not success:
            return
        key = F.normalize(key.float(), dim=0).detach().cpu()
        action = action.detach().cpu().float()
        self._age += 1
        self.entries.append(MemoryEntry(key=key, action=action, score=1.0, age=self._age))
        if len(self.entries) > self.capacity:
            self.entries.sort(key=lambda x: x.age)
            self.entries = self.entries[-self.capacity :]

    def refine(self, current_action: torch.Tensor, retrieved_action: torch.Tensor) -> torch.Tensor:
        return self.blend_alpha * current_action.float() + (1.0 - self.blend_alpha) * retrieved_action.float()
