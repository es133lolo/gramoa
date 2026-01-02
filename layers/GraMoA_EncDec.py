import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_ff,
        dropout,
        activation="relu",
        use_temporal_moa=True,
        use_spatial_moa=False,
        temporal_moa=None,
        spatial_moa=None,
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.use_temporal_moa = use_temporal_moa
        self.use_spatial_moa = use_spatial_moa

        self.temporal_moa = temporal_moa
        self.spatial_moa = spatial_moa

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv3 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        # ★ ADformerLayer가 router_latents를 반환하도록 변경됨
        new_x_t, new_x_c, attn_t, attn_c, _router_info = \
            self.attention(x_t, x_c, attn_mask=attn_mask, tau=tau, delta=delta)
        
        # 2. Temporal MoA (선택)
        if self.use_temporal_moa and self.temporal_moa is not None:
            new_x_t, _, t_latents, t_gates = self.temporal_moa(
                new_x_t, attn_mask=None, tau=tau, delta=delta
            )
        else:
            t_latents, t_gates = None, None

        # 3. Spatial MoA (선택)
        if self.use_spatial_moa and self.spatial_moa is not None:
            new_x_c, _, s_latents, s_gates = self.spatial_moa(
                new_x_c, attn_mask=None, tau=tau, delta=delta
            )
        else:
            s_latents, s_gates = None, None
            
        x_t = [_x_t + self.dropout(_nx_t) for _x_t, _nx_t in zip(x_t, new_x_t)]
        x_c = [_x_c + self.dropout(_nx_c) for _x_c, _nx_c in zip(x_c, new_x_c)]

        y_t = x_t = [self.norm1(_x_t) for _x_t in x_t]
        y_t = [self.dropout(self.activation(self.conv1(_y_t.transpose(-1, 1)))) for _y_t in y_t]
        y_t = [self.dropout(self.conv2(_y_t).transpose(-1, 1)) for _y_t in y_t]

        y_c = x_c = [self.norm3(_x_c) for _x_c in x_c]
        y_c = [self.dropout(self.activation(self.conv3(_y_c.transpose(-1, 1)))) for _y_c in y_c]
        y_c = [self.dropout(self.conv4(_y_c).transpose(-1, 1)) for _y_c in y_c]

        router_info = {
            "temporal": {
                "latents": t_latents,
                "gates": t_gates,
            },
            "spatial": {
                "latents": s_latents,
                "gates": s_gates,
            }
        }

        return (
            [self.norm2(_x_t + _y_t) for _x_t, _y_t in zip(x_t, y_t)],
            [self.norm4(_x_c + _y_c) for _x_c, _y_c in zip(x_c, y_c)],
            attn_t,
            attn_c,
            router_info
        )


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_t, x_c, attn_mask=None, tau=None, delta=None):
        # x [[B, L1, D], [B, L2, D], ...]
        attns_t = []
        attns_c = []


        all_router_info = {
            "temporal": {"latents": [], "gates": []},
            "spatial": {"latents": [], "gates": []},
        }

        for attn_layer in self.attn_layers:
            x_t, x_c, attn_t, attn_c, router_info = \
                attn_layer(x_t, x_c, attn_mask=attn_mask, tau=tau, delta=delta)
            
            # attention 기록 (output_attention 호환)
            attns_t.append(attn_t)
            attns_c.append(attn_c)

            if router_info["temporal"]["latents"] is not None:
                all_router_info["temporal"]["latents"].append(
                    router_info["temporal"]["latents"]
                )
                all_router_info["temporal"]["gates"].append(
                    router_info["temporal"]["gates"]
                )

            if router_info["spatial"]["latents"] is not None:
                all_router_info["spatial"]["latents"].append(
                    router_info["spatial"]["latents"]
                )
                all_router_info["spatial"]["gates"].append(
                    router_info["spatial"]["gates"]
                )

        for branch in ["temporal", "spatial"]:
            if len(all_router_info[branch]["latents"]) > 0:
                all_router_info[branch]["latents"] = torch.stack(
                    all_router_info[branch]["latents"], dim=0
                )  # [L, B, G, D]
                all_router_info[branch]["gates"] = torch.stack(
                    all_router_info[branch]["gates"], dim=0
                )  # [L, B, G, E]


        """# concat all the outputs
        if x_t:
            x_t = torch.cat(x_t, dim=1)  # (batch_size, patch_num_1 + patch_num_2 + ... , d_model)
        else:
            x_t = None
        if x_c:
            x_c = torch.cat(x_c, dim=1)  # (batch_size, enc_in_1 + enc_in_2 + ... , d_model)
        else:
            x_c = None"""
        # only concat the routers. router is the last patch/channel of each element in the list

        """if x_t:
            x_t = torch.cat([x[:, -1, :].unsqueeze(1) for x in x_t], dim=1)   # (batch_size, len(patch_len_list), d_model)
        else:
            x_t = None
        if x_c:
            x_c = torch.cat([x[:, -1, :].unsqueeze(1) for x in x_c], dim=1)  # (batch_size, len(up_dim_list), d_model)
        else:
            x_c = None

        if self.norm is not None:
            x_t = self.norm(x_t) if x_t is not None else None
            x_c = self.norm(x_c) if x_c is not None else None"""

        if x_t:
            x_t = [x[:, -1, :].unsqueeze(1) for x in x_t]  # [(batch_size, 1, d_model), ...]
        else:
            x_t = None
        if x_c:
            x_c = [x[:, -1, :].unsqueeze(1) for x in x_c]  # [(batch_size, 1, d_model), ...]
        else:
            x_c = None

        return x_t, x_c, attns_t, attns_c, all_router_info