import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


# SupConRouterLoss for Router Embeddings
class SupConRouterLoss(nn.Module):
    """
    Positive Pair: (subject_id, granularity_id) 동일한 경우만.
    Input:
        router_repr: [B, G, D]
        contrast_labels: [B, G] # 같은 사람, 같은 Granularity끼리 > positive pair. 같은 routing 성향 같도록 유도.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, router_repr, contrast_labels):
        B, G, D = router_repr.shape
        N = B * G
        
        router_repr = router_repr.reshape(N, D)
        labels = contrast_labels.reshape(N)
        
        # normalize
        router_repr = F.normalize(router_repr, dim=1)
        
        # similarity matrix
        sim = torch.matmul(router_repr, router_repr.T) / self.temperature
        
        # positive mask (same label, excluding self)
        label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask_self = torch.eye(N, device=router_repr.device).bool()
        mask_pos = label_eq & ~mask_self # 자신은 제외
        
        # for numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        
        exp_sim = torch.exp(sim)
        
        # denominator: all except self
        denom = exp_sim.masked_fill(mask_self, 0).sum(dim=1, keepdim=True)
        
        # log probability
        log_prob = sim - torch.log(denom + 1e-8)
        
        # mean over positive pairs per anchor
        num_pos = mask_pos.sum(dim=1)
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (num_pos + 1e-8)
        
        # only consider anchors with at least one positive
        valid_mask = num_pos > 0 # positive pair가 없는 anchor 제외
        loss = -mean_log_prob_pos[valid_mask].mean()
        
        return loss if valid_mask.any() else torch.tensor(0.0, device=router_repr.device)

class LoadBalancingLoss(nn.Module): # MoE 붕괴 방지
    """
    Computes the auxiliary load balancing loss to encourage experts to be selected uniformly.
    Reference: Switch Transformer (Fedus et al., 2021)
    """
    def __init__(self, num_experts=4):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, router_probs):
        """
        router_probs: [L, B, G, E] (GraMoA)
        """
        # [L, B, G, E] → [L*B*G, E]
        if router_probs.dim() == 4:
            router_probs = router_probs.reshape(-1, self.num_experts)
        elif router_probs.dim() == 3:
            router_probs = router_probs.reshape(-1, self.num_experts)
        else:
            raise ValueError(f"Unexpected router_probs shape: {router_probs.shape}")
        
        # Importance: 각 전문가에게 할당된 확률의 평균 (전체 토큰을 봤을 때 각 expert가 얼마나 자주 선택되나)
        importance = router_probs.mean(dim=0)

        # 값이 작을수록 전문가들이 골고루 사용됨
        # 균등 분포면 제곱합이 최소 > Hard routing이 아닌 Soft probability 기반 근사
        loss = self.num_experts * (importance ** 2).sum()
        
        return loss