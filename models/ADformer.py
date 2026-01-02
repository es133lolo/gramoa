# --- 셀 1: ADformer.py (MoA + Router 기반 분류 구조 반영) ---

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.ADformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ADformerLayer
from layers.Embed import TokenChannelEmbedding
import numpy as np


class Model(nn.Module):
    """
    ADformer + Temporal MoA 기반 모델
    - TokenChannelEmbedding: 시간/공간 multi-granularity 임베딩
    - Encoder: ADformerLayer(내부에 Temporal MoA 포함)를 e_layers 번 쌓은 인코더
    - Classifier: 각 granularity router 토큰들을 concat 후 질병 분류
    """

    def __init__(self, configs):
        super(Model, self).__init__()
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

        # 각 temporal granularity별 patch 개수 (길이 리스트)
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]

        # 나중에 참고할 수 있도록 보관
        self.patch_len_list = patch_len_list
        self.up_dim_list = up_dim_list
        self.patch_num_list = patch_num_list

        # ─────────────────────────────
        # 2. 데이터 증강 설정 (pretrain 대비)
        # ─────────────────────────────
        augmentations = configs.augmentations.split(",")
        if augmentations == ["none"] and "pretrain" in self.task_name:
            augmentations = ["flip", "frequency", "jitter", "mask", "channel", "drop"]

        # ─────────────────────────────
        # 3. Embedding: TokenChannelEmbedding
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
        # 4. Temporal MoA 하이퍼파라미터
        #    (없으면 기본값 사용)
        # ─────────────────────────────
        n_experts = getattr(configs, "n_experts", 4)
        gran_scale_init = getattr(configs, "gran_scale_init", 2.0)

        # ─────────────────────────────
        # 5. Encoder: ADformerLayer(내부에 Temporal MoA 포함)
        # ─────────────────────────────
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ADformerLayer(
                        len(patch_len_list),
                        len(up_dim_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                        n_experts=n_experts,
                        gran_scale_init=gran_scale_init,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # ─────────────────────────────
        # 6. Decoder / Classifier
        # ─────────────────────────────
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

        if self.task_name == "supervised":
            # temporal granularity 개수 = len(patch_len_list)
            num_temporal = len(patch_len_list)
            # spatial granularity 개수 = len(up_dim_list)
            num_spatial = len(up_dim_list)
            self.num_routers_total = num_temporal + num_spatial

            # 최종: 각 granularity router 토큰들을 concat → Linear 분류
            self.classifier = nn.Linear(
                configs.d_model * self.num_routers_total,
                configs.num_class,
            )

            # (선택) router representation용 projection head
            router_dim = getattr(configs, "router_dim", configs.d_model)
            self.router_proj = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model),
                nn.GELU(),
                nn.Linear(configs.d_model, router_dim),
            )

    # ─────────────────────────────
    # 7. Supervised forward (분류 + optional router_repr)
    # ─────────────────────────────
    def supervised(self, x_enc, x_mark_enc=None, return_router_repr: bool = False):
        """
        x_enc: [B, T, C]
        x_mark_enc: (현재 사용 X, 호환성 유지용)

        return_router_repr:
            False → logits만 반환 (기존 학습 코드와 완전 호환)
            True  → (logits, router_repr) 튜플 반환
                    router_repr: [B, G_t, router_dim] (temporal routers만 사용)
        """
        # 1) Embedding
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        # enc_out_t: list of [B, L_g, D], len = G_t
        # enc_out_c: list of [B, C_k, D], len = G_c

        # 2) Encoder 통과
        enc_out_t, enc_out_c, attns_t, attns_c = self.encoder(
            enc_out_t, enc_out_c, attn_mask=None
        )

        # 3) 각 temporal / spatial granularity의 router 토큰 추출 (마지막 토큰)
        routers_t = []
        routers_c = []

        if enc_out_t is not None and len(enc_out_t) > 0:
            routers_t = [x[:, -1:, :] for x in enc_out_t]  # [B, 1, D] * G_t

        if enc_out_c is not None and len(enc_out_c) > 0:
            routers_c = [x[:, -1:, :] for x in enc_out_c]  # [B, 1, D] * G_c

        if not routers_t and not routers_c:
            raise RuntimeError(
                "Both temporal and spatial blocks are disabled; "
                "at least one of them should be active."
            )

        # 모든 router를 concat → [B, G_total, D]
        routers_all = routers_t + routers_c
        enc_out = torch.cat(routers_all, dim=1)  # (B, G_total, D)

        # 4) 분류 헤드
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (B, G_total * D)
        logits = self.classifier(output)              # (B, num_classes)

        # 5) (선택) router representation 반환 (contrastive 용도)
        if return_router_repr:
            if routers_t:
                temporal_routers = torch.cat(routers_t, dim=1)  # [B, G_t, D]
                router_repr = self.router_proj(temporal_routers)  # [B, G_t, router_dim]
            else:
                router_repr = None
            return logits, router_repr

        return logits

    # ─────────────────────────────
    # 8. forward: 기존 학습 코드와 완전 호환
    # ─────────────────────────────
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        exp_supervised.py에서:
            outputs = self.model(batch_x, padding_mask, None, None)
        형태로 호출되고 있으므로,
        여기서는 logits만 반환하도록 유지한다.
        """
        if self.task_name == "supervised":
            # 기존 파이프라인과의 호환성을 위해 router_repr은 기본 False
            output = self.supervised(x_enc, x_mark_enc, return_router_repr=False)
            return output
        else:
            raise ValueError(
                "Task name not recognized or not implemented within the ADformer model"
            )
