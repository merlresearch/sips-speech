# Copyright (C) 2025-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch.nn.attention import SDPBackend, sdpa_kernel


class MultiTussModel(nn.Module):
    """Multi-channel extension of task-aware unified source separation (TUSS) [1] used in FlexIO [2].
    The current implementation takes only 'speech' prompts and supports speech separation and enhancement.
    The encoder and decoder are based on standard 2D convolution layers for 16-kHz speech.

    Parameters
    ----------
    prompts: List[str]
        List of source types. This version only supports 'speech'.
    conf_cross_prompt_module: Dict
        Configuration for the cross-prompt module.
        See .yaml config files under ./configs directory for details.
    conf_cond_tse_module: Dict
        Configuration for the conditional TSE module.
    nblocks_cross_prompt_module: int
        Number of the TF-Locoformer blocks in the cross-prompt module.
    nblocks_cond_tse_module: int
        Number of the TF-Locoformer blocks in the conditional TSE module.
    sample_rate: int
        Sample rate of the input audio.
    stft_size: int
        STFT size of the input audio.
    eps: float
        Small constant for normalization layer.
    prompt_size: int
        Number of prompt tokens. This version only supports 1.
    use_sos_token: bool
        Whether to use SOS token. If True, the model will learn the SOS token vector.
    ref_channel: int
        Reference channel index for the input mixture. This version only supports 0.
    estimation: choice of ["masking", "mapping"]
        Whether the model estimates the mask or the complex spectrogram.
    cross_prompt_module_name: str
        The type of multi-channel cross-prompt module. This version only supports "coatt".
    References
    ----------
    [1]: Kohei Saijo, Janek Ebbers, François G Germain, Gordon Wichern, Jonathan Le Roux,
    "Task-Aware Unified Source Separation," Proc. ICASSP, 2025.
    [2] Yoshiki Masuyama, Kohei Saijo, Francesco Paissan, Jiangyu Han, Marc Delcroix,
    Ryo Aihara, François G Germain, Gordon Wichern, Jonathan Le Roux, "FlexIO: Flexible
    Single- and Multi-Channel Speech Separation and Enhancement," Proc. ICASSP, 2026.
    """

    def __init__(
        self,
        prompts: List[str],
        conf_cross_prompt_module: Dict,
        conf_cond_tse_module: Dict,
        nblocks_cross_prompt_module: int = 4,
        nblocks_cond_tse_module: int = 2,
        sample_rate: int = 16000,
        stft_size: int = 1024,
        eps: float = 1.0e-5,
        prompt_size: int = 1,
        use_sos_token: bool = True,
        ref_channel: int = 0,
        estimation: str = "masking",
        cross_prompt_module_name: str = "coatt",
    ):
        super().__init__()

        self.ref_channel = ref_channel
        self.estimation = estimation

        self.cross_prompt_module = nn.ModuleList([])
        rope_freq_first = RotaryEmbedding(
            conf_cross_prompt_module["attention_dim"] // conf_cross_prompt_module["n_heads"]
        )
        rope_time_first = RotaryEmbedding(
            conf_cross_prompt_module["attention_dim"] // conf_cross_prompt_module["n_heads"]
        )
        for bidx in range(nblocks_cross_prompt_module):
            if cross_prompt_module_name == "coatt":
                self.cross_prompt_module.append(
                    CoAttTFLocoformerBlock(
                        rope_freq_first,
                        rope_time_first,
                        eps=eps,
                        **conf_cross_prompt_module,
                    )
                )
            else:
                raise NotImplementedError("Only `coatt` is supported currently")

        self.cond_tse_module = nn.ModuleList([])
        rope_freq_second = RotaryEmbedding(conf_cond_tse_module["attention_dim"] // conf_cond_tse_module["n_heads"])
        rope_time_second = RotaryEmbedding(conf_cond_tse_module["attention_dim"] // conf_cond_tse_module["n_heads"])
        for _ in range(nblocks_cond_tse_module):
            self.cond_tse_module.append(
                TFLocoformerBlock(
                    rope_freq_second,
                    rope_time_second,
                    eps=eps,
                    **conf_cond_tse_module,
                )
            )

        emb_dim = conf_cross_prompt_module["emb_dim"]

        ks, padding = 3, 1
        self.conv = nn.Sequential(
            nn.Conv2d(2, emb_dim, ks, padding=padding),
            nn.GroupNorm(1, emb_dim, eps=eps),
        )
        self.deconv = nn.ConvTranspose2d(
            emb_dim,
            2,
            ks,
            padding=padding,
        )

        self.use_sos_token = use_sos_token
        self.prompt_size = prompt_size
        self.prompts = nn.ParameterDict({})
        for prompt in prompts:
            self.prompts[prompt] = nn.Parameter(torch.randn(emb_dim, self.prompt_size, 1))

        if self.use_sos_token:
            self.sos_token = nn.Parameter(torch.randn(emb_dim, self.prompt_size, 1))

    def forward(self, input: torch.Tensor, prompts: List[List[str]]) -> torch.Tensor:
        batch0 = input.unsqueeze(-1)
        batch = torch.cat((batch0.real, batch0.imag), dim=-1)
        n_batch, n_mics, n_frames, n_freqs = batch.shape[:4]
        num_bands = n_freqs
        n_src = len(prompts[0])

        batch = rearrange(batch, "b m t f c -> (b m) c t f")
        batch = self.conv(batch)

        batch = rearrange(batch, "(b m) c t f -> b m c t f", b=n_batch, m=n_mics)

        p = self.prompts[prompts[0][0]]
        p = p.unsqueeze(0).unsqueeze(1).repeat(n_batch, n_mics, 1, n_src, num_bands)
        if self.use_sos_token:
            sos_token = self.sos_token.unsqueeze(0).repeat(n_batch, n_mics, 1, 1, num_bands)
            batch = torch.cat((p, sos_token, batch), dim=3)
        else:
            batch = torch.cat((p, batch), dim=3)

        for block in self.cross_prompt_module:
            batch = block(batch)

        prompt_vectors, batch = (
            batch[:, self.ref_channel, :, : n_src * self.prompt_size, :],
            batch[:, self.ref_channel, :, n_src * self.prompt_size :, :],
        )
        prompt_vectors = prompt_vectors.reshape(n_batch, -1, n_src, self.prompt_size, num_bands).transpose(1, 2)

        if self.use_sos_token:
            batch = batch[..., self.prompt_size :, :]

        batch = batch.unsqueeze(1).repeat(1, n_src, 1, 1, 1)
        batch = batch * prompt_vectors
        batch = batch.reshape(n_batch * n_src, -1, n_frames, num_bands)

        for block in self.cond_tse_module:
            batch = block(batch, n_src=n_src)

        batch = self.deconv(batch)

        batch = batch.view([n_batch, n_src, 2, n_frames, n_freqs])
        batch = batch.to(torch.float32)

        batch = torch.complex(batch[:, :, 0], batch[:, :, 1])

        if self.estimation == "masking":
            batch = batch0[:, self.ref_channel, ...].movedim(-1, -3) * batch
            return batch
        else:
            return batch


class TFLocoformerBlock(nn.Module):
    def __init__(
        self,
        rope_freq,
        rope_time,
        emb_dim=128,
        num_groups=4,
        tf_order="ft",
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        freq_ffn_config=[],
        frame_ffn_config=[],
        mic_ffn_config=[],
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        assert tf_order in ["tf", "ft"], tf_order
        self.tf_order = tf_order

        self.freq_path = LocoformerBlock(
            rope_freq,
            emb_dim=emb_dim,
            num_groups=num_groups,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_config=freq_ffn_config,
            dropout=dropout,
            eps=eps,
        )
        self.frame_path = LocoformerBlock(
            rope_time,
            emb_dim=emb_dim,
            num_groups=num_groups,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_config=frame_ffn_config,
            dropout=dropout,
            eps=eps,
        )

    def forward(self, input, n_src=None):
        """TF-Locoformer forward.

        input: torch.Tensor
            Input tensor, (n_batch, channel, n_frame, n_freq)
        """
        if self.tf_order == "ft":
            output = self.freq_frame_process(input)
        else:
            output = self.frame_freq_process(input)

        return output

    def freq_frame_process(self, input):
        output = input.movedim(-3, -1)  # (B, T, Q_old, H)
        output = self.freq_path(output)

        output = output.transpose(-3, -2)  # (B, F, T, H)
        output = self.frame_path(output)
        return output.transpose(-1, -3)

    def frame_freq_process(self, input):
        # Input tensor, (n_batch, hidden, n_frame, n_freq)
        output = input.transpose(-3, -1)  # (B, F, T, H)
        output = self.frame_path(output)

        output = output.transpose(-3, -2)  # (B, T, F, H)
        output = self.freq_path(output)
        return output.movedim(-1, -3)


class CoAttTFLocoformerBlock(TFLocoformerBlock):
    def __init__(
        self,
        rope_freq,
        rope_time,
        emb_dim=128,
        num_groups=4,
        tf_order="ft",
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        freq_ffn_config=[],
        frame_ffn_config=[],
        mic_ffn_config=[],
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__(
            rope_freq=rope_freq,
            rope_time=rope_time,
            emb_dim=emb_dim,
            num_groups=num_groups,
            tf_order=tf_order,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            freq_ffn_config=freq_ffn_config,
            frame_ffn_config=frame_ffn_config,
            dropout=dropout,
            eps=eps,
        )

        self.freq_path = CoAttLocoformerBlock(
            rope_freq,
            emb_dim=emb_dim,
            num_groups=num_groups,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_config=freq_ffn_config,
            dropout=dropout,
            eps=eps,
        )
        self.frame_path = CoAttLocoformerBlock(
            rope_time,
            emb_dim=emb_dim,
            num_groups=num_groups,
            n_heads=n_heads,
            flash_attention=flash_attention,
            attention_dim=attention_dim,
            ffn_config=frame_ffn_config,
            dropout=dropout,
            eps=eps,
        )


class LocoformerBlock(nn.Module):
    def __init__(
        self,
        rope,
        emb_dim=128,
        num_groups=4,
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        ffn_config=[],
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])

        assert len(ffn_config) in [1, 2], ffn_config
        self.macaron_style = len(ffn_config) == 2
        for ffn_conf in ffn_config[::-1]:
            self.ffn_norm.append(RMSGroupNorm(num_groups, emb_dim, eps=eps))
            if ffn_conf["ffn_type"] == "swiglu_conv1d":
                self.ffn.append(SwiGLUConvDeconv1d(dim=emb_dim, dropout=dropout, **ffn_conf["conf"]))
            else:
                raise ValueError

        self.attn_norm = RMSGroupNorm(num_groups, emb_dim, eps=eps)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    def forward(self, x):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either of the number of frames or freqs
        """
        B, T, F, C = x.shape

        if self.macaron_style:
            input_ = x
            output = self.ffn_norm[-1](x)
            output = self.ffn[-1](output)
            output = output + input_
        else:
            output = x

        input_ = output
        output = self.attn_norm(output)
        output = output.contiguous().view([B * T, F, C])
        output = self.attn(output)
        output = output.contiguous().view([B, T, F, C]) + input_

        input_ = output
        output = self.ffn_norm[0](output)
        output = self.ffn[0](output)
        output = output + input_

        return output


class CoAttLocoformerBlock(nn.Module):
    def __init__(
        self,
        rope,
        emb_dim=128,
        num_groups=4,
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        ffn_config=[],
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])

        assert len(ffn_config) in [1, 2], ffn_config
        self.macaron_style = len(ffn_config) == 2
        for ffn_conf in ffn_config[::-1]:
            self.ffn_norm.append(RMSGroupNorm(num_groups, emb_dim, eps=eps))
            if ffn_conf["ffn_type"] == "swiglu_conv1d":
                self.ffn.append(SwiGLUConvDeconv1d(dim=emb_dim, dropout=dropout, **ffn_conf["conf"]))
            else:
                raise ValueError(f"Unsupported FFN type: {ffn_conf['ffn_type']}")

        self.attn_norm = RMSGroupNorm(num_groups, emb_dim, eps=eps)
        self.attn = MultiHeadCoAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )

    def forward(self, x):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either of the number of frames or freqs
        """
        B, M, T, F, C = x.shape

        if self.macaron_style:
            x = x.reshape(B * M, T, F, C)
            input_ = x
            output = self.ffn_norm[-1](x)
            output = self.ffn[-1](output)
            output = output + input_
            output = output.reshape(B, M, T, F, C)
        else:
            output = x

        output = output.movedim(1, 0)
        input_ = output
        output = self.attn_norm(output)
        output = output.contiguous().view([M, B * T, F, C])
        output = self.attn(output)
        output = output.contiguous().view([M, B, T, F, C]) + input_
        output = output.movedim(0, 1).reshape(B * M, T, F, C)

        input_ = output
        output = self.ffn_norm[0](output)
        output = self.ffn[0](output)
        output = output + input_

        return output.reshape(B, M, T, F, C)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.sdpb_backend = SDPBackend.FLASH_ATTENTION
        else:
            self.sdpb_backend = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

    def forward(self, input):
        query, key, value = self.get_qkv(input)
        query, key = self.apply_rope(query, key)

        with sdpa_kernel(self.sdpb_backend):
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.amp.autocast("cuda", enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key


class MultiHeadCoAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.sdpb_backend = SDPBackend.FLASH_ATTENTION
        else:
            self.sdpb_backend = [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

    def forward(self, input):
        n_mic, n_batch, seq_len = input.shape[:3]
        query, key, value = self.get_qkv(input)
        query = query.reshape(n_mic * n_batch, self.n_heads, seq_len, -1)
        key = key.reshape(n_mic * n_batch, self.n_heads, seq_len, -1)

        if self.rope is not None:
            query, key = self.apply_rope(query, key)

        query = query.reshape(n_mic, n_batch, self.n_heads, seq_len, -1)
        key = key.reshape(n_mic, n_batch, self.n_heads, seq_len, -1)

        query = query.movedim(0, -1).reshape(n_batch, self.n_heads, seq_len, -1)
        key = key.movedim(0, -1).reshape(n_batch, self.n_heads, seq_len, -1)
        value = value.movedim(0, -1).reshape(n_batch, self.n_heads, seq_len, -1)

        with sdpa_kernel(self.sdpb_backend):
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )

        output = output.reshape(n_batch, self.n_heads, seq_len, -1, n_mic)
        output = output.contiguous().permute(-1, 0, 2, 1, 3)
        output = output.reshape(n_mic, n_batch, seq_len, -1)
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_mic, n_batch, seq_len = input.shape[:3]
        x = self.qkv(input).reshape(n_mic, n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 2)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.amp.autocast("cuda", enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner=None,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        dim_inner = dim_inner if dim_inner is not None else dim * 4

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either of the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x).view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in the vision field, RMSGroupNorm
        is applied in each TF bin.

        Args:
            num_groups: int
                Number of groups
            dim: int
                number of dimensions
            eps: float
                Small constant to avoid zero division.
            bias: bool
                Whether to add bias term. RMSNorm does not use bias.

        """
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output
