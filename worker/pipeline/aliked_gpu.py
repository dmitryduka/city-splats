"""ALIKED-N16rot on a torch device, reproducing COLMAP's ONNX extractor (aliked.cc): the
image padded to a multiple of 32 by repeating its edge, the top `max_keypoints` after NMS,
then those below `min_score` dropped, coordinates in COLMAP's convention (pixel corner at 0).
Imported only by worker.pipeline.gpu_features' subprocess (torch, no pycolmap)."""

from __future__ import annotations

import numpy as np
import torch
from lightglue.aliked import ALIKED

# The feature aggregation in one Metal pass per pixel: the 4 parts (32 channels each; parts
# 2-4 upsampled bilinearly with aligned corners), the score head's first 1x1 conv, and the
# L2-normalised 128-d feature. As separate MPS ops (upsample, cat, F.normalize, conv) it took
# ~230 ms of the ~300 ms per 1024 px image; this takes ~47 ms (features within 7e-6).
_FUSE = r"""
#include <metal_stdlib>
using namespace metal;
float bilinear(device const float* p, int ch, int hp, int wp, int H, int W, int y, int x) {
    float sy = H > 1 ? float(y) * float(hp - 1) / float(H - 1) : 0.0;
    float sx = W > 1 ? float(x) * float(wp - 1) / float(W - 1) : 0.0;
    int y0 = int(sy), x0 = int(sx);
    int y1 = min(y0 + 1, hp - 1), x1 = min(x0 + 1, wp - 1);
    float ly = sy - float(y0), lx = sx - float(x0);
    device const float* q = p + ch * hp * wp;
    return (1.0 - ly) * ((1.0 - lx) * q[y0 * wp + x0] + lx * q[y0 * wp + x1])
         + ly * ((1.0 - lx) * q[y1 * wp + x0] + lx * q[y1 * wp + x1]);
}
kernel void fuse(device const float* a, device const float* b, device const float* c, device const float* d,
                 device const float* w0, device float* feat, device float* s0, constant int* dims,
                 uint idx [[thread_position_in_grid]]) {
    int H = dims[0], W = dims[1];
    if (int(idx) >= H * W) return;
    int y = int(idx) / W, x = int(idx) % W;
    float ss = 0.0;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 128; ++k) {
        int part = k / 32, ch = k % 32;
        float v = part == 0 ? a[ch * H * W + idx]
                : part == 1 ? bilinear(b, ch, dims[2], dims[3], H, W, y, x)
                : part == 2 ? bilinear(c, ch, dims[4], dims[5], H, W, y, x)
                : bilinear(d, ch, dims[6], dims[7], H, W, y, x);
        ss += v * v;
        for (int o = 0; o < 8; ++o) acc[o] += w0[o * 128 + k] * v;
        feat[k * H * W + idx] = v;
    }
    float inv = 1.0 / max(sqrt(ss), 1e-12);
    for (int k = 0; k < 128; ++k) feat[k * H * W + idx] *= inv;
    for (int o = 0; o < 8; ++o) s0[o * H * W + idx] = acc[o];
}
"""


class AlikedGPU:
    def __init__(self, device: str, max_keypoints: int, min_score: float):
        self.device, self.min_score = device, min_score
        # detection_threshold -1: DKD keeps the top max_keypoints, as the ONNX export does
        self.net = ALIKED(model_name="aliked-n16rot", max_num_keypoints=max_keypoints,
                          detection_threshold=-1).eval().to(device)
        self.fuse = torch.mps.compile_shader(_FUSE).fuse if device == "mps" else None
        self.w0 = self.net.score_head[0].weight.reshape(8, -1).contiguous()  # 1x1, no bias

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """rgb: H x W x 3 uint8. Keypoints (N x 2, pixels) and float32 descriptors (N x 128)."""
        h, w = rgb.shape[:2]
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float() / 255
        ph, pw = -(-h // 32) * 32, -(-w // 32) * 32
        if (ph, pw) != (h, w):
            x = torch.nn.functional.pad(x, (0, pw - w, 0, ph - h), mode="replicate")
        net = self.net
        # extract_dense_map without its unpadding: the ONNX model keeps the padded maps
        x1 = net.block1(x)
        x2 = net.block2(net.pool2(x1))
        x3 = net.block3(net.pool4(x2))
        x4 = net.block4(net.pool4(x3))
        x1, x2 = net.gate(net.conv1(x1)), net.gate(net.conv2(x2))
        x3, x4 = net.gate(net.conv3(x3)), net.gate(net.conv4(x4))
        if self.fuse is not None:
            H, W = x1.shape[2:]
            feature_map = torch.empty(1, 128, H, W, device=self.device)
            s0 = torch.empty(1, 8, H, W, device=self.device)
            dims = torch.tensor([H, W, *x2.shape[2:], *x3.shape[2:], *x4.shape[2:]], dtype=torch.int32,
                                device=self.device)
            self.fuse(x1.contiguous(), x2.contiguous(), x3.contiguous(), x4.contiguous(), self.w0,
                      feature_map, s0, dims, threads=H * W)
            score_map = torch.sigmoid(net.score_head[1:](s0))
        else:
            x1234 = torch.cat([x1, net.upsample2(x2), net.upsample8(x3), net.upsample32(x4)], 1)
            score_map = torch.sigmoid(net.score_head(x1234))
            feature_map = torch.nn.functional.normalize(x1234, p=2, dim=1)
        keypoints, scores, _ = net.dkd(score_map)
        kp, score, desc = keypoints[0], scores[0], self._describe(feature_map[0], keypoints[0])
        keep = score >= self.min_score
        kp, desc = kp[keep].cpu().numpy(), desc[keep].cpu().numpy()
        px = (kp[:, 0] + 1) * 0.5 * (pw - 1) + 0.5
        py = (kp[:, 1] + 1) * 0.5 * (ph - 1) + 0.5
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        return np.stack([px, py], 1)[inside].astype(np.float32), desc[inside].astype(np.float32)

    def _describe(self, feature_map: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """SDDH.forward for one image (kernel 3, no mask, no conv2D) with the patch lookup as
        flat indexing and the 1x1 sf_conv as a matmul: the same numbers, 61 -> ~45 ms on MPS."""
        sddh = self.net.desc_head
        c, h, w = feature_map.shape
        wh = torch.tensor([[w - 1, h - 1]], device=feature_map.device)
        kp_wh = (keypoints / 2 + 0.5) * wh
        n, ps = len(kp_wh), sddh.kernel_size
        corner = (kp_wh.long() - ps / 2 + 1).long()  # get_patches
        corner[:, 0] = corner[:, 0].clamp(min=0, max=w - 1 - ps)
        corner[:, 1] = corner[:, 1].clamp(min=0, max=h - 1 - ps)
        o = torch.arange(ps, device=feature_map.device)
        index = ((corner[:, 1, None, None] + o[None, :, None]) * w + corner[:, 0, None, None] + o[None, None, :]).reshape(-1)
        patch = feature_map.reshape(c, -1)[:, index].reshape(c, n, ps, ps).permute(1, 0, 2, 3)
        offset = sddh.offset_conv(patch).clamp(-max(h, w) / 4.0, max(h, w) / 4.0)
        offset = offset[:, :, 0, 0].view(n, 2, sddh.n_pos).permute(0, 2, 1)
        pos = (2.0 * (kp_wh.unsqueeze(1) + offset) / wh[None] - 1).reshape(1, n * sddh.n_pos, 1, 2)
        features = torch.nn.functional.grid_sample(feature_map[None], pos, mode="bilinear", align_corners=True)
        features = features.reshape(c, n, sddh.n_pos).permute(1, 0, 2)  # n x c x n_pos
        features = torch.selu(torch.matmul(sddh.sf_conv.weight[:, :, 0, 0], features))
        return torch.nn.functional.normalize(torch.einsum("ncp,pcd->nd", features, sddh.agg_weights), p=2.0, dim=1)
