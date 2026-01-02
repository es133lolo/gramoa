import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.GraMoA_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import GraMoALayer, MoABlock
from layers.Embed import TokenChannelEmbedding

import numpy as np


class Model(nn.Module):
    """
    ADformer + Temporal MoA 기반 모델 (수정됨)
    
    주요 변경사항:
    - Classifier: Temporal routers만 사용 (Spatial 제거)
    - Contrastive: enc_out_t에서 router representation 추출 (alignment 일치)
    - [NEW] Router Probability 반환 기능 추가 (Load Balancing Loss용)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # MoA usage flags (for ablation)
        self.use_temporal_moa = not configs.no_temporal_moa
        self.use_spatial_moa = not configs.no_spatial_moa
        self.noisy_gating = configs.noisy_gating
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in

        # ─────────────────────────────
        # 1. Temporal / Spatial granularity 설정
        # ─────────────────────────────
        if configs.no_temporal_block and configs.no_spatial_block:
            raise ValueError("At least one of the two blocks should be True")

        # Temporal granularity (patch_len_list)
        if configs.no_temporal_block:
            patch_len_list = []
        else:
            patch_len_list = list(map(int, configs.patch_len_list.split(",")))

        # Spatial granularity (up_dim_list)
        if configs.no_spatial_block:
            up_dim_list = []
        else:
            up_dim_list = list(map(int, configs.up_dim_list.split(",")))

        stride_list = patch_len_list
        seq_len = configs.seq_len

        # 각 temporal granularity별 patch 개수
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]

        self.patch_len_list = patch_len_list
        self.up_dim_list = up_dim_list
        self.patch_num_list = patch_num_list
        self.num_temporal = len(patch_len_list)

        # ─────────────────────────────
        # 2. 데이터 증강 설정
        # ─────────────────────────────
        augmentations = configs.augmentations.split(",")
        if augmentations == ["none"] and "pretrain" in self.task_name:
            augmentations = ["flip", "frequency", "jitter", "mask", "channel", "drop"]

        # ─────────────────────────────
        # 3. Embedding
        # ─────────────────────────────
        self.enc_embedding = TokenChannelEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            patch_len_list,
            up_dim_list,
            stride_list,
            configs.dropout,
            augmentations,
        )

        # ─────────────────────────────
        # MoA blocks (shared across layers)
        # ─────────────────────────────
        self.temporal_moa = (
            MoABlock(
                n_gran=len(self.patch_len_list),
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                n_experts=configs.n_experts,
                dropout=configs.dropout,
                output_attention=configs.output_attention,
                noisy_gating=self.noisy_gating,
            )
            if self.use_temporal_moa
            else None
        )

        self.spatial_moa = (
            MoABlock(
                n_gran=len(self.up_dim_list),
                d_model=configs.d_model,
                n_heads=configs.n_heads,
                n_experts=configs.n_experts,
                dropout=configs.dropout,
                output_attention=configs.output_attention,
                noisy_gating=self.noisy_gating,
            )
            if self.use_spatial_moa
            else None
        )

        # ─────────────────────────────
        # 4. Temporal MoA 하이퍼파라미터
        # ─────────────────────────────
        n_experts = getattr(configs, "n_experts", 4)
        gran_scale_init = getattr(configs, "gran_scale_init", 1.0)

        # ─────────────────────────────
        # Attention type selection (ONCE)
        # ─────────────────────────────
        if not self.use_temporal_moa and not self.use_spatial_moa:
            # True baseline (No MoA at all)
            attention = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=configs.dropout,
                    output_attention=configs.output_attention,
                ),
                configs.d_model,
                configs.n_heads,
            )
        else:
            # GraMoA attention (Temporal / Spatial / Dual)
            attention = GraMoALayer(
                len(patch_len_list),
                len(up_dim_list),
                configs.d_model,
                configs.n_heads,
                configs.dropout,
                configs.output_attention,
                configs.no_inter_attn,
                n_experts=n_experts,
                gran_scale_init=gran_scale_init,
                noisy_gating=True,
            )



        # ─────────────────────────────
        # 5. Encoder
        # ─────────────────────────────
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention=attention,
                    d_model=configs.d_model,
                    d_ff=configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    use_temporal_moa=self.use_temporal_moa,
                    use_spatial_moa=self.use_spatial_moa,
                    temporal_moa=self.temporal_moa,
                    spatial_moa=self.spatial_moa,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # ─────────────────────────────
        # 6. Classifier & Projection Head
        # ─────────────────────────────
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

        if self.task_name == "supervised":
            active_branches = int(self.use_temporal_moa) + int(self.use_spatial_moa)

            d_model = configs.d_model
            
            if active_branches == 0:
                clf_in_dim = d_model
            else:
                clf_in_dim = d_model * active_branches

            self.classifier = nn.Linear(clf_in_dim, configs.num_class)

            
            # ★ [수정 1] Router Projection Head에 Dropout 추가!
            # 과적합의 주범인 대조 학습(Contrastive Learning) 파트의 규제를 강화
            router_dim = getattr(configs, "router_dim", configs.d_model)
            self.router_proj = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.Dropout(configs.dropout),
                nn.GELU(),
                nn.Linear(configs.d_model, router_dim),
            )

    # ─────────────────────────────
    # 7. Supervised forward (수정됨)
    # ─────────────────────────────
    def supervised(self, x_enc, x_mark_enc=None, return_router_repr: bool = False, return_router_probs: bool = False):
        """
        Args:
            return_router_repr: Contrastive Loss용 표현 반환 여부
            return_router_probs: Load Balancing Loss용 router gate 확률, shape: [L, B, G, E]

        """
        # 1) Embedding
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)

        # 2) Encoder 통과
        enc_out_t, enc_out_c, attns_t, attns_c, router_info = self.encoder(
            enc_out_t, enc_out_c, attn_mask=None
        )

        reprs = []

        if self.use_temporal_moa:
            # enc_out_t: list of [B, N_i, D]
            # → 각 granularity에서 router token (마지막 토큰)만 사용
            temporal_repr = torch.stack(
                [x[:, -1, :] for x in enc_out_t], dim=1
            ).mean(dim=1)   # [B, D]
            reprs.append(temporal_repr)

        if self.use_spatial_moa:
            spatial_repr = torch.stack(
                [x[:, -1, :] for x in enc_out_c], dim=1
            ).mean(dim=1)   # [B, D]
            reprs.append(spatial_repr)

        if not self.use_temporal_moa and not self.use_spatial_moa:
            global_repr = torch.cat(enc_out_t, dim=1).mean(dim=1)
            reprs = [global_repr]

        assert len(reprs) > 0, "No representation selected for classifier"

        final_repr = torch.cat(reprs, dim=1)
        output = self.act(final_repr)
        output = self.dropout(output)
        logits = self.classifier(output)

        router_repr = None

        if return_router_repr and self.use_temporal_moa:
            # Temporal alignment only
            routers_t = torch.cat(enc_out_t, dim=1)   # [B, G_t, D]
            router_repr = self.router_proj(routers_t)



        # 5) 반환값 구성
        results = [logits]
      
        if return_router_repr:
            results.append(router_repr)

        if return_router_probs:
            # Load Balancing Loss 계산을 위해 확률값router_probs 반환
            results.append(router_info)

        if len(results) == 1:
            return results[0]
        else:
            return tuple(results)

    # ─────────────────────────────
    # 8. forward (수정됨)
    # ─────────────────────────────
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_router_repr=False, return_router_probs=False):
        """
        기존 학습 코드와 호환되는 forward 인터페이스
        """
        if self.task_name == "supervised":
            return self.supervised(
                x_enc, 
                x_mark_enc, 
                return_router_repr=return_router_repr, 
                return_router_probs=return_router_probs  # ★ 인자 전달
            )
        else:
            raise ValueError("Task name not recognized")