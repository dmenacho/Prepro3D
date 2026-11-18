"""2D teacher features for 3D points: SAM2 / SAM3 semantic feature map -> AnyUp (image-guided) -> per-point features.

Why these feature maps (checked in the official code):
  * SAM2.1 Hiera-L (1024x1024 input): `image_encoder(...)["vision_features"]` is the stride-16 FPN level,
    64x64x256, the only fine level with top-down semantic fusion (`fpn_top_down_levels: [2, 3]`); its stride-4/8
    levels are lateral projections of early backbone stages (edges/texture, little semantics).
  * SAM3 (1008x1008 input): `backbone.forward_image(...)["vision_features"]` is the 72x72x256 neck level. Its
    288x288 level is only a transposed-conv expansion of the same 72x72 ViT tokens, so it adds no image detail.
Both models stretch the full image to a square (no padding), so pixel (u, v) of the original image maps to the
same relative position in their feature maps. AnyUp then upsamples the coarse map using the full-resolution image,
the role LoftUp plays for DINOv2 in Vernata; the upsampled map is sampled bilinearly at each projected point.
"""

import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
ANYUP_CKPT = "https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth"


def _to_tensor(image_uint8, device):
    """HxWx3 uint8 numpy -> 1x3xHxW float in [0, 1]."""
    return torch.from_numpy(image_uint8).to(device).permute(2, 0, 1)[None].float() / 255.0


class SAM2Features:
    name, resolution, dim = "sam2", 1024, 256

    def __init__(self, device, checkpoint=None, model_id="facebook/sam2.1-hiera-large", random_weights=False):
        from sam2.build_sam import build_sam2, build_sam2_hf
        cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        if random_weights:
            self.model = build_sam2(cfg, ckpt_path=None, device=device)
        elif checkpoint:
            self.model = build_sam2(cfg, ckpt_path=checkpoint, device=device)
        else:
            self.model = build_sam2_hf(model_id, device=device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image_uint8):
        x = F.interpolate(_to_tensor(image_uint8, self.device), size=(self.resolution,) * 2, mode="bilinear", antialias=True, align_corners=False)
        x = (x - IMAGENET_MEAN.to(x)) / IMAGENET_STD.to(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = self.model.image_encoder(x)["vision_features"]
        return feats.float()                                     # 1 x 256 x 64 x 64


class SAM3Features:
    name, resolution, dim = "sam3", 1008, 256

    def __init__(self, device, checkpoint=None, random_weights=False):
        from sam3.model_builder import build_sam3_image_model
        self.model = build_sam3_image_model(
            device=device, eval_mode=True, checkpoint_path=checkpoint,
            load_from_HF=(checkpoint is None and not random_weights))
        self.device = device

    @torch.no_grad()
    def __call__(self, image_uint8):
        x = F.interpolate(_to_tensor(image_uint8, self.device), size=(self.resolution,) * 2, mode="bilinear", antialias=True, align_corners=False)
        x = (x - 0.5) / 0.5                                      # SAM3 normalisation (mean = std = 0.5)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = self.model.backbone.forward_image(x)["vision_features"]
        return feats.float()                                     # 1 x 256 x 72 x 72


def build_featurizer(backbone, device, checkpoint=None, random_weights=False):
    if backbone == "sam2":
        return SAM2Features(device, checkpoint=checkpoint, random_weights=random_weights)
    if backbone == "sam3":
        return SAM3Features(device, checkpoint=checkpoint, random_weights=random_weights)
    raise ValueError(backbone)


class Upsampler:
    """'anyup': image-guided AnyUp upsampling to `scale` x the original image size; 'bilinear': no upsampler."""

    # NATTEN's kernel crashes (illegal memory access) at ~3M output pixels (2016x1512, tested) but works at 0.76M
    # (1008x756). Above this limit AnyUp uses its standard attention (same weights; much slower and more memory).
    NATTEN_MAX_PIXELS = 1_000_000

    def __init__(self, kind, device, scale=0.5, q_chunk_size=None, use_natten=True):
        self.kind, self.device, self.scale, self.q_chunk_size = kind, device, scale, q_chunk_size
        self.use_natten, self._models, self._warned = use_natten, {}, False
        if kind not in ("anyup", "bilinear"):
            raise ValueError(kind)

    def _anyup(self, natten):
        if natten not in self._models:
            from anyup.model import AnyUp
            m = AnyUp().to(self.device)
            m.load_state_dict(torch.hub.load_state_dict_from_url(ANYUP_CKPT, progress=False, map_location=self.device))
            if natten:   # same as the official hubconf `anyup_multi_backbone(use_natten=True)`
                from anyup.layers import setup_cross_attention_block
                cd = m.cross_decode
                m.cross_decode = setup_cross_attention_block(
                    use_natten=True, qk_dim=cd.cross_attn.attention.embed_dim, num_heads=4,
                    window_ratio=cd.window_ratio, use_params_from=cd).to(self.device)
            self._models[natten] = m.eval()
        return self._models[natten]

    @torch.no_grad()
    def __call__(self, image_uint8, lr_feats, uv, width, height):
        """Per-point features (N x C float32 numpy) at original-image pixel coordinates uv (N x 2)."""
        if self.kind == "anyup":
            H, W = int(round(height * self.scale)), int(round(width * self.scale))
            img = F.interpolate(_to_tensor(image_uint8, self.device), size=(H, W), mode="bilinear", antialias=True, align_corners=False)
            img = (img - IMAGENET_MEAN.to(img)) / IMAGENET_STD.to(img)
            natten = self.use_natten and H * W <= self.NATTEN_MAX_PIXELS
            if self.use_natten and not natten and not self._warned:
                print(f"AnyUp: {W}x{H} output > {self.NATTEN_MAX_PIXELS} pixels -> standard attention instead of NATTEN "
                      "(slower, more GPU memory)", flush=True)
                self._warned = True
            q_chunk = self.q_chunk_size if natten else (self.q_chunk_size or 65536)
            fmap = self._anyup(natten)(img, lr_feats, output_size=(H, W), q_chunk_size=q_chunk)
        else:
            fmap = lr_feats
        uv_t = torch.as_tensor(uv, dtype=torch.float32, device=self.device)
        grid = torch.stack([(uv_t[:, 0] + 0.5) / width * 2 - 1, (uv_t[:, 1] + 0.5) / height * 2 - 1], dim=-1)
        out = F.grid_sample(fmap, grid.view(1, 1, -1, 2), mode="bilinear", align_corners=False)
        return out[0, :, 0].T.contiguous().cpu().numpy()
