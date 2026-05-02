# -*- coding: utf-8 -*-
"""
D-CeLR: Dual-encoder Cephalometric Landmark Regression (PyTorch)

This module implements the full model described in:
"A Cephalometric Landmark Regression Method based on Dual-encoder for High-resolution X-ray Image"

Pipeline (see Fig. 2–3 in the paper):
  1) Feature extractor (ResNet34) -> multi-level S2..S5 -> per-level 1x1 proj to d_model
     + 2D positional maps -> fused feature Fu = F2+F3+F4+F5
     + auxiliary heatmap head from S5 (Dice + MSE in training).
  2) Reference encoder (Transformer encoder only):
     - tokens = [K landmark content queries || S5 image tokens]
     - add learned token-type embeddings + position embeddings
     - outputs coarse landmark tokens -> FFN -> (mu_R, sigma_R)
  3) Finetune encoder (Transformer encoder, M layers):
     - initialize coords = detach(mu_R)
     - for i in 1..M:
          sample Fu at coords  -> add to landmark queries
          tokens = [updated landmark queries || Fu image tokens]
          transformer layer -> updated landmark tokens
          FFN -> (delta_i, sigma_i); coords += delta_i
     - outputs per-layer (mu_A_i, sigma_A_i) and final (mu_A, sigma_A)

Notes:
 - Coordinates (mu) are in pixel units of the input image (H, W).
 - Feature sampling internally maps pixel coords to Fu’s normalized grid.
 - The module returns predictions; you can compute the losses externally.
"""

from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
import timm


# -----------------------------
# Positional encodings & helpers
# -----------------------------

def sine_cosine_position_embedding_2d(d_model: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Standard 2D sine-cos positional embedding (shape: [1, d_model, h, w])
    Half dims encode x, half encode y.
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sine/cos 2D embedding.")
    pe = torch.zeros(1, d_model, h, w, device=device)
    d_half = d_model // 2

    # y (rows)
    y_pos = torch.arange(h, device=device).unsqueeze(1)  # [h,1]
    div_term_y = torch.exp(torch.arange(0, d_half, 2, device=device) * -(math.log(10000.0) / d_half))
    pe_y = torch.zeros(h, d_half, device=device)
    pe_y[:, 0::2] = torch.sin(y_pos * div_term_y)
    pe_y[:, 1::2] = torch.cos(y_pos * div_term_y)
    pe_y = pe_y.unsqueeze(-1).repeat(1, 1, w)  # [h, d_half, w]

    # x (cols)
    x_pos = torch.arange(w, device=device).unsqueeze(1)  # [w,1]
    div_term_x = torch.exp(torch.arange(0, d_half, 2, device=device) * -(math.log(10000.0) / d_half))
    pe_x = torch.zeros(w, d_half, device=device)
    pe_x[:, 0::2] = torch.sin(x_pos * div_term_x)
    pe_x[:, 1::2] = torch.cos(x_pos * div_term_x)
    pe_x = pe_x.t().unsqueeze(0).repeat(h, 1, 1)  # [h, d_half, w]

    pe[:, 0:d_half, :, :] = pe_y.permute(1, 0, 2)  # [d_half,h,w]
    pe[:, d_half:, :, :] = pe_x.permute(1, 0, 2)
    return pe  # [1, d_model, h, w]


def flatten_hw_to_seq(feat: torch.Tensor) -> torch.Tensor:
    """
    [B, C, H, W] -> [B, H*W, C]
    """
    b, c, h, w = feat.shape
    return feat.flatten(2).transpose(1, 2)


def normalize_pixel_coords_to_grid(coords_xy: torch.Tensor, H_img: int, W_img: int, H_feat: int, W_feat: int) -> torch.Tensor:
    """
    coords_xy: [B, K, 2] in pixel units of the input image (x, y)
    Map to normalized grid in feature map space Fu of size [H_feat, W_feat] (range [-1,1]).
    """
    # Map to feature coordinates (not normalized yet)
    stride_y = H_img / float(H_feat)
    stride_x = W_img / float(W_feat)

    x_feat = coords_xy[..., 0] / stride_x  # [B,K]
    y_feat = coords_xy[..., 1] / stride_y

    # Normalize to [-1,1] for grid_sample
    x_norm = (x_feat / (W_feat - 1)) * 2.0 - 1.0
    y_norm = (y_feat / (H_feat - 1)) * 2.0 - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)  # [B,K,2]
    return grid


def bilinear_sample_at_points(feat: torch.Tensor, coords_xy_imgspace: torch.Tensor, H_img: int, W_img: int) -> torch.Tensor:
    """
    Sample features at image-pixel coords on fused feature map feat.
    feat: [B, C, Hf, Wf]
    coords_xy_imgspace: [B, K, 2] (x,y) in pixel units of the input image
    returns: [B, K, C]
    """
    b, c, hf, wf = feat.shape
    grid = normalize_pixel_coords_to_grid(coords_xy_imgspace, H_img, W_img, hf, wf)  # [B,K,2]
    grid = grid.unsqueeze(2)  # [B,K,1,2]
    # grid_sample expects [B,C,H_out,W_out] with grid [B,H_out,W_out,2]
    sampled = F.grid_sample(feat, grid, mode='bilinear', align_corners=True)  # [B,C,K,1]
    sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()  # [B,K,C]
    return sampled


def image_patches_at_points(
    image: torch.Tensor,
    coords_xy_imgspace: torch.Tensor,
    patch_size: int,
    radius_px: float,
) -> torch.Tensor:
    """
    Crop differentiable square image patches centered at landmark coordinates.
    Returns [B*K, C, patch_size, patch_size].
    """
    b, c, h, w = image.shape
    k = coords_xy_imgspace.shape[1]
    device = image.device
    offsets = torch.linspace(-radius_px, radius_px, patch_size, device=device)
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")

    centers = coords_xy_imgspace.reshape(b * k, 2)
    sample_x = centers[:, 0].view(-1, 1, 1) + xx.view(1, patch_size, patch_size)
    sample_y = centers[:, 1].view(-1, 1, 1) + yy.view(1, patch_size, patch_size)
    x_norm = (sample_x / max(w - 1, 1)) * 2.0 - 1.0
    y_norm = (sample_y / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1)

    patches = []
    for idx in range(k):
        start = idx * b
        end = start + b
        patches.append(
            F.grid_sample(
                image,
                grid[start:end],
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        )
    return torch.cat(patches, dim=0)


# -----------------------------
# Losses (auxiliary helpers)
# -----------------------------

def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Multi-channel dice loss on heatmaps.
    logits: [B, K, H, W] (unnormalized)
    targets: [B, K, H, W] (0/1 or soft)
    """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    dice = 1.0 - (num + eps) / den
    return dice.mean()


def rle_loss_laplace(mu_pred: torch.Tensor, log_sigma: torch.Tensor, mu_gt: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """
    Residual Log-likelihood Estimation (Laplace form commonly used in keypoint regression):
       L = |mu - mu_gt|_2 * exp(-s) + s
    where s = log_sigma (scalar per keypoint).
    Shapes:
      mu_pred:   [B, K, 2]
      log_sigma: [B, K, 1]
      mu_gt:     [B, K, 2]
    """
    # radial L2 distance per keypoint
    residual = torch.linalg.norm(mu_pred - mu_gt, dim=-1, keepdim=True)  # [B,K,1]
    loss = residual * torch.exp(-log_sigma) + log_sigma
    return loss.mean() if reduce else loss


# -----------------------------
# Feature Extractor (ResNet34)
# -----------------------------

class ResNet34MultiFeature(nn.Module):
    """
    Returns S2, S3, S4, S5 from ResNet-34.
    If in_channels=1, first conv is adapted for grayscale.
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT
        backbone = resnet34(weights=weights)

        if in_channels != 3:
            # replace conv1 for non-RGB
            conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                # if grayscale, average pretrained RGB weights
                if in_channels == 1:
                    conv1.weight[:] = backbone.conv1.weight.mean(dim=1, keepdim=True)
                else:
                    # random init if channels not 1 or 3
                    nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
            backbone.conv1 = conv1

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # -> S2
        self.layer2 = backbone.layer2  # -> S3
        self.layer3 = backbone.layer3  # -> S4
        self.layer4 = backbone.layer4  # -> S5

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)      # /4
        s2 = self.layer1(x)   # /4
        s3 = self.layer2(s2)  # /8
        s4 = self.layer3(s3)  # /16
        s5 = self.layer4(s4)  # /32
        return s2, s3, s4, s5


class ConvNeXtV2MultiFeature(nn.Module):
    """
    ConvNeXt V2 backbone (base) returning four stages with strides 4/8/16/32.
    """
    def __init__(self, in_channels: int = 1, variant: str = "convnextv2_base"):
        super().__init__()
        # features_only=True returns list of feature maps at out_indices
        self.backbone = timm.create_model(
            variant,
            features_only=True,
            pretrained=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3),
        )
        info = self.backbone.feature_info
        self.channels = info.channels()  # list of C for each out index

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        # feats is a list: strides ~ [4,8,16,32]
        if len(feats) != 4:
            raise RuntimeError(f"Expected 4 feature maps from ConvNeXtV2 backbone, got {len(feats)}")
        return tuple(feats)  # type: ignore[return-value]


# -----------------------------
# Dual-Encoder D-CeLR
# -----------------------------

class DCeLR(nn.Module):
    """
    Dual-encoder Cephalometric Landmark Regressor (D-CeLR).
    """
    def __init__(
        self,
        num_landmarks: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers_ref: int = 4,
        num_layers_finetune: int = 4,
        ff_dim: int = 2048,
        in_channels: int = 1,
        heatmap_channels: Optional[int] = None,  # if None -> K
        dropout: float = 0.1,
        backbone: str = "convnextv2_base",
        use_fpn_refine: bool = False,
        fpn_dim: int = 256,
        fpn_refine_level: str = "p2",
        use_patch_refine: bool = False,
        patch_size: int = 64,
        patch_radius: float = 32.0,
        use_hr_heatmap_refine: bool = False,
    ):
        super().__init__()
        K = num_landmarks
        self.K = K
        self.d_model = d_model
        self.in_channels = in_channels
        self.use_fpn_refine = use_fpn_refine
        self.fpn_refine_level = fpn_refine_level
        self.use_patch_refine = use_patch_refine
        self.patch_size = patch_size
        self.patch_radius = patch_radius
        self.use_hr_heatmap_refine = use_hr_heatmap_refine
        self.use_fpn_features = use_fpn_refine or use_hr_heatmap_refine

        # 1) Feature extractor -----------------------------------------------
        self.backbone_name = backbone
        if backbone == "resnet34":
            self.backbone = ResNet34MultiFeature(in_channels=in_channels)
            channels = (64, 128, 256, 512)
        elif backbone.startswith("convnextv2"):
            convnext = ConvNeXtV2MultiFeature(in_channels=in_channels, variant=backbone)
            self.backbone = convnext
            channels = convnext.channels
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'. Use 'resnet34' or convnextv2 variants.")

        c2, c3, c4, c5 = channels
        self.proj_s2 = nn.Conv2d(c2, d_model, kernel_size=1)
        self.proj_s3 = nn.Conv2d(c3, d_model, kernel_size=1)
        self.proj_s4 = nn.Conv2d(c4, d_model, kernel_size=1)
        self.proj_s5 = nn.Conv2d(c5, d_model, kernel_size=1)

        # Heatmap head from S5 (auxiliary)
        self.heatmap_out = nn.Conv2d(c5, heatmap_channels or K, kernel_size=1)

        # Optional high-resolution FPN point refiner. The transformer still
        # attends over S5-sized tokens; the FPN is sampled only at landmark
        # coordinates to keep memory bounded at 1024px inputs.
        if self.use_fpn_features:
            if fpn_refine_level not in {"p2", "p3", "p4", "multi"}:
                raise ValueError("fpn_refine_level must be one of: p2, p3, p4, multi.")
            self.fpn_lateral2 = nn.Conv2d(c2, fpn_dim, kernel_size=1)
            self.fpn_lateral3 = nn.Conv2d(c3, fpn_dim, kernel_size=1)
            self.fpn_lateral4 = nn.Conv2d(c4, fpn_dim, kernel_size=1)
            self.fpn_lateral5 = nn.Conv2d(c5, fpn_dim, kernel_size=1)
            self.fpn_smooth2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.fpn_smooth3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
            self.fpn_smooth4 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

        if self.use_fpn_refine:
            if fpn_refine_level == "multi":
                self.local_fpn_project = nn.Sequential(
                    nn.LayerNorm(fpn_dim * 3),
                    nn.Linear(fpn_dim * 3, fpn_dim),
                    nn.GELU(),
                )
            self.local_landmark_embed = nn.Parameter(torch.randn(K, fpn_dim) * 0.02)
            self.local_offset_head = nn.Sequential(
                nn.LayerNorm(fpn_dim),
                nn.Linear(fpn_dim, fpn_dim),
                nn.GELU(),
                nn.Linear(fpn_dim, fpn_dim // 2),
                nn.GELU(),
                nn.Linear(fpn_dim // 2, 2),
            )
            self.local_log_sigma_head = nn.Sequential(
                nn.LayerNorm(fpn_dim),
                nn.Linear(fpn_dim, fpn_dim // 2),
                nn.GELU(),
                nn.Linear(fpn_dim // 2, 1),
            )
            nn.init.zeros_(self.local_offset_head[-1].weight)
            nn.init.zeros_(self.local_offset_head[-1].bias)
            nn.init.zeros_(self.local_log_sigma_head[-1].weight)
            nn.init.zeros_(self.local_log_sigma_head[-1].bias)

        if self.use_patch_refine:
            patch_dim = 128
            self.patch_encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(64, patch_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.patch_landmark_embed = nn.Parameter(torch.randn(K, patch_dim) * 0.02)
            self.patch_offset_head = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, patch_dim),
                nn.GELU(),
                nn.Linear(patch_dim, 2),
            )
            self.patch_log_sigma_head = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, patch_dim // 2),
                nn.GELU(),
                nn.Linear(patch_dim // 2, 1),
            )
            nn.init.zeros_(self.patch_offset_head[-1].weight)
            nn.init.zeros_(self.patch_offset_head[-1].bias)
            nn.init.zeros_(self.patch_log_sigma_head[-1].weight)
            nn.init.zeros_(self.patch_log_sigma_head[-1].bias)

        if self.use_hr_heatmap_refine:
            self.hr_heatmap_head = nn.Conv2d(fpn_dim, K, kernel_size=1)
            self.hr_offset_head = nn.Conv2d(fpn_dim, K * 2, kernel_size=1)
            self.hr_log_sigma_head = nn.Sequential(
                nn.LayerNorm(fpn_dim),
                nn.Linear(fpn_dim, fpn_dim // 2),
                nn.GELU(),
                nn.Linear(fpn_dim // 2, 1),
            )
            self.hr_fusion_logit = nn.Parameter(torch.full((K, 1), -2.0))
            nn.init.zeros_(self.hr_offset_head.weight)
            nn.init.zeros_(self.hr_offset_head.bias)
            nn.init.zeros_(self.hr_log_sigma_head[-1].weight)
            nn.init.zeros_(self.hr_log_sigma_head[-1].bias)

        # 2) Reference encoder ------------------------------------------------
        enc_layer_ref = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=False
        )
        self.reference_encoder = nn.TransformerEncoder(enc_layer_ref, num_layers=num_layers_ref)

        # Landmark content queries (learned) for reference encoder
        self.landmark_queries_ref = nn.Parameter(torch.randn(K, d_model))

        # Token type embeddings: 0 -> landmark, 1 -> image
        self.token_type_embed = nn.Embedding(2, d_model)

        # Heads to predict coarse coordinates & log_sigma from landmark tokens
        self.head_coarse_mu = nn.Linear(d_model, 2)
        self.head_coarse_log_sigma = nn.Linear(d_model, 1)

        # 3) Finetune encoder -------------------------------------------------
        # A stack of M independent transformer encoder layers (layer-by-layer updating)
        self.finetune_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=False
            ) for _ in range(num_layers_finetune)
        ])

        # Landmark content queries (learned) for finetune stage
        self.landmark_queries_fine = nn.Parameter(torch.randn(K, d_model))

        # Per-layer heads for delta & sigma
        self.head_delta = nn.ModuleList([nn.Linear(d_model, 2) for _ in range(num_layers_finetune)])
        self.head_fine_log_sigma = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_layers_finetune)])

        # LayerNorm to stabilize additions
        self.ln_landmark = nn.LayerNorm(d_model)
        self.ln_image = nn.LayerNorm(d_model)

    # --------- Feature extractor with fusion and positional maps ---------

    def _feature_extractor(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
          {
            'S2','S3','S4','S5': raw backbone features (for debugging/vis),
            'F2'..'F5': projected + positional,
            'Fu': fused d_model feature,
            'Hmap': auxiliary heatmap logits from S5
          }
        """
        b, _, H, W = x.shape
        s2, s3, s4, s5 = self.backbone(x)  # S2..S5

        # Auxiliary heatmap from raw S5 (512 channels)
        hmap_logits = self.heatmap_out(s5)

        # Project each to d_model
        p2 = self.proj_s2(s2)
        p3 = self.proj_s3(s3)
        p4 = self.proj_s4(s4)
        p5 = self.proj_s5(s5)

        # Resize to S5 spatial size and add 2D positional maps
        _, _, h5, w5 = p5.shape
        device = x.device
        pos5 = sine_cosine_position_embedding_2d(self.d_model, h5, w5, device)

        def to_s5_and_add_pos(p: torch.Tensor) -> torch.Tensor:
            t = F.interpolate(p, size=(h5, w5), mode='bilinear', align_corners=True)
            return t + pos5  # broadcast over batch

        f2 = to_s5_and_add_pos(p2)
        f3 = to_s5_and_add_pos(p3)
        f4 = to_s5_and_add_pos(p4)
        f5 = p5 + pos5

        fu = f2 + f3 + f4 + f5  # fused feature

        fpn_refine = None
        fpn_pyramid = None
        if self.use_fpn_features:
            p5 = self.fpn_lateral5(s5)
            p4 = self.fpn_lateral4(s4) + F.interpolate(p5, size=s4.shape[-2:], mode='bilinear', align_corners=True)
            p3 = self.fpn_lateral3(s3) + F.interpolate(p4, size=s3.shape[-2:], mode='bilinear', align_corners=True)
            p2 = self.fpn_lateral2(s2) + F.interpolate(p3, size=s2.shape[-2:], mode='bilinear', align_corners=True)
            p4 = self.fpn_smooth4(p4)
            p3 = self.fpn_smooth3(p3)
            p2 = self.fpn_smooth2(p2)
            fpn_pyramid = {"p2": p2, "p3": p3, "p4": p4}
            if self.fpn_refine_level == "multi":
                fpn_refine = fpn_pyramid
            else:
                fpn_refine = fpn_pyramid[self.fpn_refine_level]

        return dict(
            S2=s2,
            S3=s3,
            S4=s4,
            S5=s5,
            F2=f2,
            F3=f3,
            F4=f4,
            F5=f5,
            Fu=fu,
            Hmap=hmap_logits,
            FPNRefine=fpn_refine,
            FPNPyramid=fpn_pyramid,
        )

    # --------- Reference encoder (coarse) ---------

    def _reference_stage(
        self,
        s5_proj_pos: torch.Tensor,  # [B, d_model, h5, w5] (that's F5)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build tokens = [K landmark queries || image tokens], run encoder, return:
        - coarse landmark tokens, coarse mu, coarse log_sigma
        """
        b, c, h5, w5 = s5_proj_pos.shape
        K = self.K
        # Image tokens from S5
        img_seq = flatten_hw_to_seq(s5_proj_pos)  # [B, L, d], L=h5*w5

        # Landmark content queries (same for each batch), expand to [B,K,d]
        lmk_queries = self.landmark_queries_ref.unsqueeze(0).expand(b, K, c)

        # Add token-type embeddings
        type_lmk = self.token_type_embed.weight[0].unsqueeze(0).unsqueeze(0)  # [1,1,d]
        type_img = self.token_type_embed.weight[1].unsqueeze(0).unsqueeze(0)  # [1,1,d]
        lmk_tok = lmk_queries + type_lmk
        img_tok = img_seq + type_img

        # Concatenate and Transformer-encode (convert to [S,B,E])
        tokens = torch.cat([lmk_tok, img_tok], dim=1)            # [B, K+L, d]
        tokens = tokens.transpose(0, 1).contiguous()             # [K+L, B, d]
        enc_out = self.reference_encoder(tokens)                 # [K+L, B, d]
        enc_out = enc_out.transpose(0, 1).contiguous()           # [B, K+L, d]

        # Landmark slice
        lmk_out = enc_out[:, :K, :]                              # [B,K,d]

        # Heads -> coarse mu, log_sigma
        mu_R = self.head_coarse_mu(lmk_out)                      # [B,K,2]
        log_sigma_R = self.head_coarse_log_sigma(lmk_out)        # [B,K,1]

        return lmk_out, mu_R, log_sigma_R

    # --------- Finetune encoder (iterative) ---------

    def _finetune_stage(
        self,
        fu: torch.Tensor,             # [B, d_model, hf, wf]
        img_tokens_fu: torch.Tensor,  # [B, L, d_model]
        mu_init: torch.Tensor,        # [B, K, 2] (detached mu_R)
        H_img: int,
        W_img: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Iteratively refine coordinates with M transformer layers; returns:
          mu_list:       list of [B,K,2] at each layer (mu after update)
          logsig_list:   list of [B,K,1] per layer
          mu_final:      [B,K,2]
          log_sigma_final: [B,K,1]
        """
        b, c, hf, wf = fu.shape
        K = self.K
        d = self.d_model

        # Prepare image tokens (normalized)
        img_tok = self.ln_image(img_tokens_fu + self.token_type_embed.weight[1].unsqueeze(0).unsqueeze(0))  # [B,L,d]

        # Initialize landmark queries (same for each batch)
        lmk_queries = self.landmark_queries_fine.unsqueeze(0).expand(b, K, d)  # [B,K,d]

        # Start from detached coarse coords
        mu = mu_init.detach()

        mu_list: List[torch.Tensor] = []
        logsig_list: List[torch.Tensor] = []

        for layer_idx, (layer, head_d, head_s) in enumerate(zip(self.finetune_layers, self.head_delta, self.head_fine_log_sigma)):
            # Sample fused features at current coords (image space -> Fu)
            sampled = bilinear_sample_at_points(fu, mu, H_img=H_img, W_img=W_img)  # [B,K,d_model] because Fu is d_model channels

            # Add sampled features to the content queries
            lmk_tok = self.ln_landmark(lmk_queries + sampled + self.token_type_embed.weight[0].unsqueeze(0).unsqueeze(0))  # [B,K,d]

            # Concatenate with image tokens and pass one transformer encoder layer
            tokens = torch.cat([lmk_tok, img_tok], dim=1)     # [B, K+L, d]
            tokens = tokens.transpose(0, 1).contiguous()      # [K+L, B, d]
            out = layer(tokens)                               # one layer
            out = out.transpose(0, 1).contiguous()            # [B, K+L, d]

            # Updated landmark tokens
            lmk_out = out[:, :K, :]                           # [B,K,d]

            # Heads: delta and sigma (this layer)
            delta = head_d(lmk_out)                           # [B,K,2]
            log_sigma_i = head_s(lmk_out)                     # [B,K,1]

            # Update coords
            mu = mu + delta

            mu_list.append(mu)
            logsig_list.append(log_sigma_i)

            # Optional: carry the updated landmark tokens forward (content memory)
            lmk_queries = lmk_out

        mu_final = mu
        log_sigma_final = logsig_list[-1]
        return mu_list, logsig_list, mu_final, log_sigma_final

    def _local_offset_stage(
        self,
        fpn_feat,
        mu_init: torch.Tensor,
        H_img: int,
        W_img: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict a final coordinate residual from high-resolution FPN features
        sampled at the current landmark locations.
        """
        b = mu_init.shape[0]
        if isinstance(fpn_feat, dict):
            sampled_levels = [
                bilinear_sample_at_points(fpn_feat[level], mu_init, H_img=H_img, W_img=W_img)
                for level in ("p2", "p3", "p4")
            ]
            sampled = self.local_fpn_project(torch.cat(sampled_levels, dim=-1))
        else:
            sampled = bilinear_sample_at_points(fpn_feat, mu_init, H_img=H_img, W_img=W_img)
        landmark_embed = self.local_landmark_embed.unsqueeze(0).expand(b, -1, -1)
        local_feat = sampled + landmark_embed
        delta = self.local_offset_head(local_feat)
        log_sigma = self.local_log_sigma_head(local_feat)
        return mu_init + delta, log_sigma, delta

    def _patch_refine_stage(
        self,
        image: torch.Tensor,
        mu_init: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Refine each landmark from a high-resolution image patch centered on
        the current coordinate estimate.
        """
        b, k, _ = mu_init.shape
        patches = image_patches_at_points(
            image=image,
            coords_xy_imgspace=mu_init,
            patch_size=self.patch_size,
            radius_px=self.patch_radius,
        )
        patch_feat = self.patch_encoder(patches).flatten(1).view(b, k, -1)
        patch_feat = patch_feat + self.patch_landmark_embed.unsqueeze(0)
        delta = self.patch_offset_head(patch_feat)
        log_sigma = self.patch_log_sigma_head(patch_feat)
        return mu_init + delta, log_sigma, delta

    def _hr_heatmap_stage(
        self,
        p2: torch.Tensor,
        H_img: int,
        W_img: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dense stride-4 heatmap + offset coordinate head.
        The heatmap gives the expected cell location and the offset head
        predicts the sub-cell correction in input-image pixels.
        """
        b, c, h, w = p2.shape
        heatmap_logits = self.hr_heatmap_head(p2)
        offset = self.hr_offset_head(p2).view(b, self.K, 2, h, w)
        probs = heatmap_logits.flatten(2).softmax(dim=-1).view(b, self.K, h, w)

        grid_y = torch.arange(h, device=p2.device, dtype=p2.dtype).view(1, 1, h, 1)
        grid_x = torch.arange(w, device=p2.device, dtype=p2.dtype).view(1, 1, 1, w)
        x_feat = (probs * grid_x).sum(dim=(2, 3))
        y_feat = (probs * grid_y).sum(dim=(2, 3))
        stride_x = W_img / float(w)
        stride_y = H_img / float(h)
        base = torch.stack([(x_feat + 0.5) * stride_x, (y_feat + 0.5) * stride_y], dim=-1)

        offset_xy = (probs.unsqueeze(2) * offset).sum(dim=(3, 4))
        mu = base + offset_xy
        weighted_feat = (probs.unsqueeze(2) * p2.unsqueeze(1)).sum(dim=(3, 4))
        log_sigma = self.hr_log_sigma_head(weighted_feat)
        return mu, log_sigma, heatmap_logits, offset, probs

    # --------- Forward ---------

    def forward(
        self,
        x: torch.Tensor,
        gt_coords: Optional[torch.Tensor] = None,
        gt_heatmap: Optional[torch.Tensor] = None,
        loss_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> Dict[str, torch.Tensor]:
        """
        x: [B, C=1 or 3, H, W]
        gt_coords:  optional [B, K, 2] in pixel units (for computing losses here if desired)
        gt_heatmap: optional [B, K, Hs5, Ws5] aligned to S5 for aux loss
        loss_weights: (lambda_HM, lambda_RE, lambda_FE)

        Returns dict with predictions (and losses if gt provided):
          - aux_heatmap: [B, K, Hs5, Ws5]
          - coarse_mu, coarse_log_sigma: [B,K,2], [B,K,1]
          - fine_mu_per_layer: list len M of [B,K,2]
          - fine_log_sigma_per_layer: list len M of [B,K,1]
          - fine_mu, fine_log_sigma: final [B,K,2], [B,K,1]
          - (optional) losses: total_loss, loss_hm, loss_re, loss_fe
        """
        b, _, H, W = x.shape
        feats = self._feature_extractor(x)

        # Reference (coarse) stage
        # Use F5 (projected + positional) for image tokens in reference stage
        ref_lmk_tokens, mu_R, log_sigma_R = self._reference_stage(feats["F5"])

        # Finetune stage: image tokens come from Fu
        fu = feats["Fu"]                                # [B, d, hf, wf]
        img_tok_fu = flatten_hw_to_seq(fu)             # [B, L, d]
        mu_list, logsig_list, mu_A, log_sigma_A = self._finetune_stage(
            fu=fu, img_tokens_fu=img_tok_fu, mu_init=mu_R, H_img=H, W_img=W
        )

        local_delta = None
        if self.use_fpn_refine:
            fpn_refine = feats["FPNRefine"]
            if fpn_refine is None:
                raise RuntimeError("FPN refinement was enabled but no FPN feature was produced.")
            mu_A, log_sigma_A, local_delta = self._local_offset_stage(
                fpn_feat=fpn_refine,
                mu_init=mu_A,
                H_img=H,
                W_img=W,
            )
            mu_list.append(mu_A)
            logsig_list.append(log_sigma_A)

        patch_delta = None
        if self.use_patch_refine:
            mu_A, log_sigma_A, patch_delta = self._patch_refine_stage(x, mu_A)
            mu_list.append(mu_A)
            logsig_list.append(log_sigma_A)

        hr_mu = None
        hr_log_sigma = None
        hr_heatmap = None
        hr_offset = None
        if self.use_hr_heatmap_refine:
            fpn_pyramid = feats["FPNPyramid"]
            if fpn_pyramid is None:
                raise RuntimeError("High-resolution heatmap refinement requires FPN features.")
            hr_mu, hr_log_sigma, hr_heatmap, hr_offset, hr_probs = self._hr_heatmap_stage(
                fpn_pyramid["p2"],
                H_img=H,
                W_img=W,
            )
            fusion_w = torch.sigmoid(self.hr_fusion_logit).unsqueeze(0)
            mu_A = (1.0 - fusion_w) * mu_A + fusion_w * hr_mu
            log_sigma_A = (1.0 - fusion_w) * log_sigma_A + fusion_w * hr_log_sigma
            mu_list.append(mu_A)
            logsig_list.append(log_sigma_A)

        out: Dict[str, torch.Tensor] = {
            "aux_heatmap": feats["Hmap"],                 # [B,K,Hs5,Ws5]
            "coarse_mu": mu_R,                            # [B,K,2]
            "coarse_log_sigma": log_sigma_R,              # [B,K,1]
            "fine_mu": mu_A,                              # [B,K,2]
            "fine_log_sigma": log_sigma_A,                # [B,K,1]
        }
        if local_delta is not None:
            out["local_delta"] = local_delta
        if patch_delta is not None:
            out["patch_delta"] = patch_delta
        if hr_mu is not None:
            out["hr_mu"] = hr_mu
            out["hr_log_sigma"] = hr_log_sigma
            out["hr_heatmap"] = hr_heatmap
            out["hr_offset"] = hr_offset

        # Pack lists (for loss over layers)
        # (keep as lists for easy per-layer RLE sums)
        out["fine_mu_per_layer"] = torch.stack(mu_list, dim=0)           # [M,B,K,2]
        out["fine_log_sigma_per_layer"] = torch.stack(logsig_list, dim=0)  # [M,B,K,1]

        # Optional losses here (so the module is "complete" out-of-the-box)
        if (gt_coords is not None) or (gt_heatmap is not None):
            lam_hm, lam_re, lam_fe = loss_weights
            loss_hm = torch.tensor(0.0, device=x.device)
            loss_re = torch.tensor(0.0, device=x.device)
            loss_fe = torch.tensor(0.0, device=x.device)

            if gt_heatmap is not None:
                # Aux heatmap supervision (Dice + MSE) on S5-sized heatmaps
                # Align prediction to GT spatial size if needed
                ph, pw = feats["Hmap"].shape[-2:]
                gh, gw = gt_heatmap.shape[-2:]
                pred_hmap = feats["Hmap"]
                if (ph != gh) or (pw != gw):
                    pred_hmap = F.interpolate(pred_hmap, size=(gh, gw), mode='bilinear', align_corners=True)
                loss_hm = dice_loss_from_logits(pred_hmap, gt_heatmap) + F.mse_loss(torch.sigmoid(pred_hmap), gt_heatmap)

            if gt_coords is not None:
                # Reference encoder RLE (Eq. 3)
                loss_re = rle_loss_laplace(mu_R, log_sigma_R, gt_coords)

                # Finetune encoder RLE sum over layers (Eq. 4)
                loss_layers = []
                M = out["fine_mu_per_layer"].shape[0]
                for i in range(M):
                    mu_i = out["fine_mu_per_layer"][i]                # [B,K,2]
                    logsig_i = out["fine_log_sigma_per_layer"][i]     # [B,K,1]
                    loss_layers.append(rle_loss_laplace(mu_i, logsig_i, gt_coords, reduce=True))
                loss_fe = torch.stack(loss_layers).sum()

            total_loss = lam_hm * loss_hm + lam_re * loss_re + lam_fe * loss_fe
            out.update(dict(
                loss_total=total_loss,
                loss_hm=loss_hm,
                loss_re=loss_re,
                loss_fe=loss_fe
            ))

        return out


# -----------------------------
# Factory/helper
# -----------------------------

def build_dcelr(
    num_landmarks: int = 19,
    d_model: int = 512,
    nhead: int = 8,
    num_layers_ref: int = 4,
    num_layers_finetune: int = 4,
    ff_dim: int = 2048,
    in_channels: int = 1,
    dropout: float = 0.1,
    backbone: str = "convnextv2_base",
    use_fpn_refine: bool = False,
    fpn_dim: int = 256,
    fpn_refine_level: str = "p2",
    use_patch_refine: bool = False,
    patch_size: int = 64,
    patch_radius: float = 32.0,
    use_hr_heatmap_refine: bool = False,
):
    """
    Convenience builder with stronger default backbone (ConvNeXt V2 Base).
    Pass backbone=\"resnet34\" to match the original paper.
    """
    return DCeLR(
        num_landmarks=num_landmarks,
        d_model=d_model,
        nhead=nhead,
        num_layers_ref=num_layers_ref,
        num_layers_finetune=num_layers_finetune,
        ff_dim=ff_dim,
        in_channels=in_channels,
        heatmap_channels=num_landmarks,
        dropout=dropout,
        backbone=backbone,
        use_fpn_refine=use_fpn_refine,
        fpn_dim=fpn_dim,
        fpn_refine_level=fpn_refine_level,
        use_patch_refine=use_patch_refine,
        patch_size=patch_size,
        patch_radius=patch_radius,
        use_hr_heatmap_refine=use_hr_heatmap_refine,
    )


if __name__ == "__main__":
    # Quick sanity check (no training)
    B, C, H, W = 2, 1, 1024, 1024
    K = 19
    x = torch.randn(B, C, H, W)

    model = build_dcelr(num_landmarks=K, in_channels=C)
    out = model(x)

    print("aux_heatmap:", out["aux_heatmap"].shape)
    print("coarse_mu:", out["coarse_mu"].shape, "coarse_log_sigma:", out["coarse_log_sigma"].shape)
    print("fine_mu:", out["fine_mu"].shape, "fine_log_sigma:", out["fine_log_sigma"].shape)
    print("fine_mu_per_layer:", out["fine_mu_per_layer"].shape, "fine_log_sigma_per_layer:", out["fine_log_sigma_per_layer"].shape)
