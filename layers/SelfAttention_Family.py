import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class FullAttention(nn.Module): # Scaled Dot-Product Attention을 그대로 구현
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # Scaled Dot-Product Attention
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # multi-head
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class AttentionExpert(nn.Module): # MoA에서 쓰는 expert
    """
    하나의 temporal expert:
    - 내부적으로 기존과 동일한 FullAttention 기반 self-attention을 수행
    - ADformerLayer에서 여러 개를 만들어 MoA로 혼합
    """
    def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
        super().__init__()
        self.attn = AttentionLayer(
            FullAttention(
                False,
                factor=1,
                attention_dropout=dropout,
                output_attention=output_attention,
            ),
            d_model,
            n_heads,
            d_keys=None,
            d_values=None,
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # self-attention (Q=K=V=x)
        out, attn = self.attn(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
        )
        return out, attn


class MoABlock(nn.Module):
    """
    Temporal branch 전용 Mixture-of-Attention 블록

    입력:
        x_t: 길이 G_t(granularity 개수)인 리스트, 각 원소는 [B, N_i(시퀀스 길이), D]
    출력:
        x_out_t: MoA가 적용된 리스트 (shape는 입력과 동일)
        attn_out_t: 각 granularity별 attention (또는 None)
        router_latents: [B, G_t, D] - MoA router의 latent embeddings
    """
    def __init__(
        self,
        n_gran,
        d_model,
        n_heads,
        n_experts=4,
        dropout=0.1,
        output_attention=False,
        gran_scale_init=1.0,
        noisy_gating=False,
        noise_std=0.1,              # Noisy gating scale
        router_dropout=0.1,         # Router MLP dropout
        gate_dropout=0.0,           # Gate(logits) dropout (선택)
        return_router_gates=True,   # gates 반환 여부
    ):
        super().__init__()
        self.n_gran = n_gran
        self.n_experts = n_experts
        self.output_attention = output_attention
        self.noisy_gating = noisy_gating

        self.noise_std = float(noise_std)
        self.return_router_gates = return_router_gates

        self.router_dropout = nn.Dropout(router_dropout)
        self.gate_dropout = nn.Dropout(gate_dropout)

        # Granularity별 전용 LayerNorm, granularity마다 입력 분포가 다름
        self.gran_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_gran)]
        )
        
        # Granularity ID 임베딩
        self.gran_embed = nn.Embedding(n_gran, d_model)
        
        # Granularity embedding용 LayerNorm 추가 
        self.gran_embed_ln = nn.LayerNorm(d_model)
        
        # Granularity embedding scale (learnable)
        self.gran_scale = nn.Parameter(
            torch.tensor(gran_scale_init, dtype=torch.float32)
        )

        # Router MLP: (x_pooled + gran_emb) -> latent router embedding
        self.router_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        # Gate: latent -> expert logits
        self.gate = nn.Linear(d_model, n_experts)

        # Shared temporal experts, granularity마다 다른 expert를 고르지만 expert 자체는 공유
        self.experts = nn.ModuleList(
            [
                AttentionExpert(
                    d_model,
                    n_heads,
                    dropout=dropout,
                    output_attention=output_attention,
                )
                for _ in range(n_experts)
            ]
        )

    def forward(self, x_t, attn_mask=None, tau=None, delta=None):
        # x_t: list of [B, N_i, D]
        if len(x_t) == 0:
            # 기존 반환 형태 유지가 필요하면 None 추가로 맞추세요.
            # (다음 단계에서 EncDec까지 전파하며 통일할 예정)
            if self.return_router_gates:
                return x_t, [], None, None
            return x_t, [], None
        
        if attn_mask is None:
            attn_mask = [None] * len(x_t)

        device = x_t[0].device
        B = x_t[0].shape[0]
        gran_ids = torch.arange(len(x_t), device=device, dtype=torch.long)

        x_out = []
        attn_out = []

        router_latents = []   # [B, D] per gran # granularity별 latent router embedding 저장
        router_gates = []      # [B, E] per gran

        for g, (x_g, mask_g) in enumerate(zip(x_t, attn_mask)):
            # per-gran LayerNorm
            x_norm = self.gran_norms[g](x_g)          # [B, N_g, D]

            # Router 입력: temporal pooling + granularity embedding
            pooled = x_norm.mean(dim=1)               # [B, D]
            g_id = gran_ids[g].expand(B)              # [B]
            gran_emb_raw = self.gran_embed(g_id)      # [B, D]
            
            # gran_embed에 LayerNorm 적용 
            gran_emb = self.gran_embed_ln(gran_emb_raw)  # [B, D]

            router_input = pooled + self.gran_scale * gran_emb  # [B, D]
            
            # MLP를 통과시켜 h 정의 (contrastive loss에 쓰이는 latent)
            h = self.router_mlp(router_input)
            
            # contrastive latent h 생성
            h = self.router_dropout(h)       
            router_latents.append(h)                

            logits = self.gate(h) # expert 선호 점수
            
            # Gate dropout (expert overconfidence 방지)
            logits = self.gate_dropout(logits)
            
            # Noisy Gating 로직 적용
            # 학습 모드이고 noisy_gating이 True일 때만 노이즈 추가
            if self.training and self.noisy_gating:
                noise = torch.randn_like(logits) * (self.noise_std / self.n_experts)
                # 필요에 따라 noise scale 조절 가능
                logits = logits + noise

            gates = torch.softmax(logits, dim=-1)     # [B, E]

            router_gates.append(gates)
            
            # 모든 expert 통과
            expert_outs = []
            expert_attns = [] if self.output_attention else None

            for expert in self.experts:
                y_e, attn_e = expert(
                    x_norm, attn_mask=mask_g, tau=tau, delta=delta
                )
                expert_outs.append(y_e.unsqueeze(-1))  # [B, N_g, D, 1]
                if self.output_attention:
                    expert_attns.append(attn_e.unsqueeze(-1))

            expert_outs = torch.cat(expert_outs, dim=-1)        # [B, N_g, D, E]
            gates_exp = gates.unsqueeze(1).unsqueeze(1)         # [B, 1, 1, E]
            y_mix = (expert_outs * gates_exp).sum(dim=-1)       # [B, N_g, D]

            # attention도 필요하면 expert별로 가중합
            if self.output_attention and expert_attns and expert_attns[0] is not None:
                attn_stack = torch.cat(expert_attns, dim=-1)    # [B, H, N_g, N_g, E]
                gates_attn = gates.unsqueeze(1).unsqueeze(1).unsqueeze(1)  # [B,1,1,1,E]
                attn_mix = (attn_stack * gates_attn).sum(dim=-1)          # [B, H, N_g, N_g]
            else:
                attn_mix = None

            x_out.append(y_mix)
            attn_out.append(attn_mix)

        # router_latents → (B, G_t, D)
        router_latents = torch.stack(router_latents, dim=1)

        # router_gates 리스트를 텐서로 변환
        router_gates = torch.stack(router_gates, dim=1)     # [B, G, E]
        # 4개 반환
        return x_out, attn_out, router_latents, router_gates

TemporalMoABlock = MoABlock
SpatialMoABlock  = MoABlock

class GraMoALayer(nn.Module): # Temporal / Spatial MoA를 완전히 대칭 구조로 감싸기
    def __init__(
        self,
        num_blocks_t,
        num_blocks_c,
        d_model,
        n_heads,
        dropout=0.1,
        output_attention=False,
        no_inter=False,
        n_experts=4,
        gran_scale_init=1.0,
        noisy_gating=False,
        use_temporal_moa=True,
        use_spatial_moa=False,
    ):
        super().__init__()

        self.use_temporal_moa = use_temporal_moa
        self.use_spatial_moa  = use_spatial_moa

        # ========== Temporal branch ==========
        if self.use_temporal_moa:
            self.temporal_moa = TemporalMoABlock(
                n_gran=num_blocks_t,
                d_model=d_model,
                n_heads=n_heads,
                n_experts=n_experts,
                dropout=dropout,
                output_attention=output_attention,
                gran_scale_init=gran_scale_init,
                noisy_gating=noisy_gating,
            )
            self.intra_attentions_t = None
        else:
            self.temporal_moa = None
            self.intra_attentions_t = nn.ModuleList([
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks_t)
            ])

        # ========== Spatial branch ==========
        if self.use_spatial_moa:
            self.spatial_moa = SpatialMoABlock(
                n_gran=num_blocks_c,
                d_model=d_model,
                n_heads=n_heads,
                n_experts=n_experts,
                dropout=dropout,
                output_attention=output_attention,
                gran_scale_init=gran_scale_init,
                noisy_gating=noisy_gating,
            )
            self.intra_attentions_c = None
        else:
            self.spatial_moa = None
            self.intra_attentions_c = nn.ModuleList([
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks_c)
            ])

        # ========== Inter-attention ==========
        self.inter_attention_t = None if no_inter or num_blocks_t <= 1 else AttentionLayer(
            FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
            d_model, n_heads
        )

        self.inter_attention_c = None if no_inter or num_blocks_c <= 1 else AttentionLayer(
            FullAttention(False, factor=1, attention_dropout=dropout, output_attention=output_attention),
            d_model, n_heads
        )

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        attn_mask_t = [None] * len(x_t)
        attn_mask_c = [None] * len(x_c)

        # ===== Temporal branch =====
        if self.use_temporal_moa:
            x_intra_t, attn_out_t, t_latents, t_gates = self.temporal_moa(
                x_t, attn_mask=attn_mask_t, tau=tau, delta=delta
            )
        else:
            x_intra_t, attn_out_t = [], []
            for x, layer, mask in zip(x_t, self.intra_attentions_t, attn_mask_t):
                y, attn = layer(x, x, x, attn_mask=mask, tau=tau, delta=delta)
                x_intra_t.append(y)
                attn_out_t.append(attn)
            t_latents, t_gates = None, None

        # ===== Spatial branch =====
        if self.use_spatial_moa:
            x_intra_c, attn_out_c, s_latents, s_gates = self.spatial_moa(
                x_c, attn_mask=attn_mask_c, tau=tau, delta=delta
            )
        else:
            x_intra_c, attn_out_c = [], []
            for x, layer, mask in zip(x_c, self.intra_attentions_c, attn_mask_c):
                y, attn = layer(x, x, x, attn_mask=mask, tau=tau, delta=delta)
                x_intra_c.append(y)
                attn_out_c.append(attn)
            s_latents, s_gates = None, None

        # ===== Inter-attention (unchanged) =====
        x_out_t = x_intra_t
        x_out_c = x_intra_c

        router_info = {
            "temporal": {"latents": t_latents, "gates": t_gates},
            "spatial":  {"latents": s_latents, "gates": s_gates},
        }

        return x_out_t, x_out_c, attn_out_t, attn_out_c, router_info
    

class ADformerLayer(nn.Module):
    def __init__(
        self,
        num_blocks_t,
        num_blocks_c,
        d_model,
        n_heads,
        dropout=0.1,
        output_attention=False,
        no_inter=False,
    ):
        super().__init__()

        self.intra_attentions_t = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks_t)
            ]
        )
        self.intra_attentions_c = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks_c)
            ]
        )
        if no_inter or num_blocks_t <= 1:
            # print("No inter attention for time")
            self.inter_attention_t = None
        else:
            self.inter_attention_t = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                n_heads,
            )
        if no_inter or num_blocks_c <= 1:
            # print("No inter attention for channel")
            self.inter_attention_c = None
        else:
            self.inter_attention_c = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                n_heads,
            )

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        attn_mask_t = ([None] * len(x_t))
        attn_mask_c = ([None] * len(x_c))

        # Intra attention
        x_intra_t = []
        attn_out_t = []
        x_intra_c = []
        attn_out_c = []
        # Temporal dimension
        for x_in_t, layer_t, mask in zip(x_t, self.intra_attentions_t, attn_mask_t):
            _x_out_t, _attn_t = layer_t(x_in_t, x_in_t, x_in_t, attn_mask=mask, tau=tau, delta=delta)
            x_intra_t.append(_x_out_t)  # (B, Ni, D)
            attn_out_t.append(_attn_t)
        # Channel dimension
        for x_in_c, layer_c, mask in zip(x_c, self.intra_attentions_c, attn_mask_c):
            _x_out_c, _attn_c = layer_c(x_in_c, x_in_c, x_in_c, attn_mask=mask, tau=tau, delta=delta)
            x_intra_c.append(_x_out_c)  # (B, C, D)
            attn_out_c.append(_attn_c)

        # Inter attention
        if self.inter_attention_t is not None:
            # Temporal dimension
            routers_t = torch.cat([x[:, -1:] for x in x_intra_t], dim=1)  # (B, n, D)
            x_inter_t, attn_inter_t = self.inter_attention_t(
                routers_t, routers_t, routers_t, attn_mask=None, tau=tau, delta=delta
            )
            x_out_t = [
                torch.cat([x[:, :-1], x_inter_t[:, i : i + 1]], dim=1)  # (B, Ni, D)
                for i, x in enumerate(x_intra_t)
            ]
            attn_out_t += [attn_inter_t]
        else:
            x_out_t = x_intra_t

        if self.inter_attention_c is not None:
            # Channel dimension
            routers_c = torch.cat([x[:, -1:] for x in x_intra_c], dim=1)  # (B, n, D)
            x_inter_c, attn_inter_c = self.inter_attention_c(
                routers_c, routers_c, routers_c, attn_mask=None, tau=tau, delta=delta
            )
            x_out_c = [
                torch.cat([x[:, :-1], x_inter_c[:, i : i + 1]], dim=1)  # (B, C, D)
                for i, x in enumerate(x_intra_c)
            ]
            attn_out_c += [attn_inter_c]
        else:
            x_out_c = x_intra_c

        return x_out_t, x_out_c, attn_out_t, attn_out_c


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(
        self, configs, seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1
    ):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(
            FullAttention(
                False,
                configs.factor,
                attention_dropout=configs.dropout,
                output_attention=configs.output_attention,
            ),
            d_model,
            n_heads,
        )
        self.dim_sender = AttentionLayer(
            FullAttention(
                False,
                configs.factor,
                attention_dropout=configs.dropout,
                output_attention=configs.output_attention,
            ),
            d_model,
            n_heads,
        )
        self.dim_receiver = AttentionLayer(
            FullAttention(
                False,
                configs.factor,
                attention_dropout=configs.dropout,
                output_attention=configs.output_attention,
            ),
            d_model,
            n_heads,
        )
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
        self.MLP2 = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # print('x', x.shape)
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        # print('time_in', time_in.shape)
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        # print('dim_in', dim_in.shape)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(
            dim_in, "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model", b=batch
        )
        # print('dim_send', dim_send.shape)
        batch_router = repeat(
            self.router,
            "seg_num factor d_model -> (repeat seg_num) factor d_model",
            repeat=batch,
        )
        # print('batch_router', batch_router.shape)
        dim_buffer, attn = self.dim_sender(
            batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None
        )
        # print('dim_buffer', dim_buffer.shape)
        dim_receive, attn = self.dim_receiver(
            dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None
        )
        # dim_receive, attn = self.dim_receiver(dim_send, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        # print('dim_receive', dim_receive.shape)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)
        # print('dim_enc', dim_enc.shape)
        # quit()

        final_out = rearrange(
            dim_enc, "(b seg_num) ts_d d_model -> b ts_d seg_num d_model", b=batch
        )

        return final_out

class MedformerLayer(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        n_heads,
        dropout=0.1,
        output_attention=False,
        no_inter=False,
    ):
        super().__init__()

        self.intra_attentions = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        if no_inter or num_blocks <= 1:
            # print("No inter attention for time")
            self.inter_attention = None
        else:
            self.inter_attention = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                n_heads,
            )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))
        # Intra attention
        x_intra = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_intra.append(_x_out)  # (B, Li, D)
            attn_out.append(_attn)
        if self.inter_attention is not None:
            # Inter attention
            routers = torch.cat([x[:, -1:] for x in x_intra], dim=1)  # (B, N, D)
            x_inter, attn_inter = self.inter_attention(
                routers, routers, routers, attn_mask=None, tau=tau, delta=delta
            )
            x_out = [
                torch.cat([x[:, :-1], x_inter[:, i : i + 1]], dim=1)  # (B, Li, D)
                for i, x in enumerate(x_intra)
            ]
            attn_out += [attn_inter]
        else:
            x_out = x_intra
        return x_out, attn_out

class FormerLayer(nn.Module):
    def __init__(self, num_blocks, d_model, n_heads, dropout=0.1, output_attention=False):
        super().__init__()

        self.intra_attentions = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks)
            ]
        )


    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))

        x_out = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_out.append(_x_out)  # (B, Li, D)
            attn_out.append(_attn)

        return x_out, attn_out


class DifferenceAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DifferenceAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(
            torch.softmax(scale * scores, dim=-1)
        )  # Scaled Dot-Product Attention
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class DifferenceAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(DifferenceAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # multi-head
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class DifferenceFormerlayer(nn.Module):
    def __init__(self, enc_in, num_blocks, d_model, n_heads, dropout=0.1, output_attention=False):
        super(DifferenceFormerlayer, self).__init__()
        self.intra_attentions = nn.ModuleList(
            [
                DifferenceAttentionLayer(
                    DifferenceAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))

        x_out = []
        attn_out = []
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_out.append(_x_out)  # (B, Li, D)
            attn_out.append(_attn)

        return x_out, attn_out