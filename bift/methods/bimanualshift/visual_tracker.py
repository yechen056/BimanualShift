import logging
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from PIL import Image

from bift.methods.bimanualshift.trajectory_gpt2 import GPT2Model


class SkillManager(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_matrix=None,
        voxel_dim=128,
        lang_dim=128,
        hidden_size=128,
        output_dim=18,
        max_voxels=8000,
        max_lang_tokens=77,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_head=4,
            n_ctx=1077,
        )

        self.max_voxels = max_voxels
        self.max_lang_tokens = max_lang_tokens
        self.embed_voxel = nn.Linear(voxel_dim, hidden_size)
        self.embed_lang = nn.Linear(lang_dim, hidden_size)
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_logits = nn.Linear(hidden_size, output_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes
        if embedding_matrix is not None:
            self.embeddings_matrix = embedding_matrix.to(self.device)

    def forward(self, voxel_embedding, language_embedding):
        batch_size = voxel_embedding.shape[0]
        voxel_embeddings = self.embed_voxel(voxel_embedding)
        language_embeddings = self.embed_lang(language_embedding)
        voxel_embeddings = voxel_embeddings.permute(0, 2, 1)
        voxel_embeddings = F.avg_pool1d(voxel_embeddings, kernel_size=16, stride=16)
        voxel_embeddings = voxel_embeddings.permute(0, 2, 1)
        inputs = torch.cat([language_embeddings, voxel_embeddings], dim=1)
        stacked_inputs = self.embed_ln(inputs)
        attention_mask = torch.ones(
            (batch_size, self.max_lang_tokens + self.max_voxels),
            device=voxel_embedding.device,
            dtype=torch.long,
        )
        assert torch.isfinite(attention_mask).all(), "attention_mask contains NaN or Inf"
        assert torch.all((attention_mask == 1)), "attention_mask contains values not equal to 1"
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=None,
        )

        hidden_state = transformer_outputs.last_hidden_state
        aggregated_hidden = hidden_state.mean(dim=1)
        logits = self.predict_logits(aggregated_hidden)
        probs = F.softmax(logits, dim=1)
        skill = torch.matmul(probs, self.embeddings_matrix.to(probs.device))
        skill = skill.view(-1, 77, 512)
        return skill


class VisualTracker(nn.Module):
    """BimanualShift-style visual tracker wrapper.

    Strict mode: when enabled, Grounded-SAM must initialize and run successfully.
    Any initialization/runtime failure raises an exception instead of fallback.
    """

    def __init__(
        self,
        backend: str = "grounded_sam",
        enabled: bool = True,
        repo_root: Optional[str] = None,
        device: str = "cuda",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        prompt: str = "robotic arm.",
        grounding_config: Optional[str] = None,
        grounding_checkpoint: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        sam_version: str = "vit_h",
        bert_base_uncased_path: Optional[str] = None,
    ):
        super().__init__()
        self.backend = backend
        self.enabled = enabled
        self.repo_root = repo_root
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.prompt = prompt
        self.grounding_config = grounding_config
        self.grounding_checkpoint = grounding_checkpoint
        self.sam_checkpoint = sam_checkpoint
        self.sam_version = sam_version
        self.bert_base_uncased_path = bert_base_uncased_path
        self._runtime_rgb = None
        self._runtime_lang = None
        self._warned_once = False
        self._initialized = False
        self._gsam_ready = False
        self._gdino_model = None
        self._sam_predictor = None
        self._gdino_transform = None

    def set_runtime_context(self, rgb=None, lang=None):
        self._runtime_rgb = rgb
        self._runtime_lang = lang

    def _warn_once(self, msg: str):
        if not self._warned_once:
            logging.warning(msg)
            self._warned_once = True

    def _lazy_init_grounded_sam(self):
        if self._initialized:
            return
        self._initialized = True
        if not self.repo_root:
            raise RuntimeError("VisualTracker: grounded_sam repo_root is empty.")
        if not os.path.isdir(self.repo_root):
            raise RuntimeError(f"VisualTracker: grounded_sam repo_root does not exist: {self.repo_root}")

        try:
            if self.repo_root not in sys.path:
                sys.path.append(self.repo_root)
            gdino_root = os.path.join(self.repo_root, "GroundingDINO")
            sam_root = os.path.join(self.repo_root, "segment_anything")
            if gdino_root not in sys.path:
                sys.path.append(gdino_root)
            if sam_root not in sys.path:
                sys.path.append(sam_root)

            import GroundingDINO.groundingdino.datasets.transforms as T  # type: ignore
            from GroundingDINO.groundingdino.models import build_model  # type: ignore
            from GroundingDINO.groundingdino.util.slconfig import SLConfig  # type: ignore
            from GroundingDINO.groundingdino.util.utils import clean_state_dict  # type: ignore
            from grounded_sam_demo import get_grounding_output  # type: ignore
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore

            cfg = self.grounding_config or os.path.join(
                self.repo_root, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            )
            gdino_ckpt = self.grounding_checkpoint or os.path.join(
                self.repo_root, "groundingdino_swint_ogc.pth"
            )
            sam_ckpt = self.sam_checkpoint or os.path.join(
                self.repo_root, "sam_vit_h_4b8939.pth"
            )

            args = SLConfig.fromfile(cfg)
            args.device = self.device
            if self.bert_base_uncased_path:
                args.bert_base_uncased_path = self.bert_base_uncased_path

            gdino = build_model(args)
            checkpoint = torch.load(gdino_ckpt, map_location="cpu")
            gdino.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            gdino.eval()

            sam = sam_model_registry[self.sam_version](checkpoint=sam_ckpt).to(self.device)
            predictor = SamPredictor(sam)

            self._gdino_model = gdino
            self._sam_predictor = predictor
            self._get_grounding_output = get_grounding_output
            self._gdino_transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self._gsam_ready = True
        except Exception as exc:
            self._gsam_ready = False
            raise RuntimeError(f"VisualTracker: failed to initialize grounded_sam ({exc})") from exc

    def _fallback_masks(self, ins: torch.Tensor):
        b, n, c = ins.shape
        side = int(round(float(n) ** (1.0 / 3.0)))
        if side * side * side != n:
            # If sequence does not map cleanly to a cube, use a simple split.
            split = n // 2
            right_mask = torch.zeros((b, n, c), device=ins.device, dtype=ins.dtype)
            right_mask[:, split:, :] = 1.0
            left_mask = 1.0 - right_mask
            return ins * right_mask, ins * left_mask

        base = torch.zeros((1, side, side, side, 1), device=ins.device, dtype=ins.dtype)
        base[:, side // 2 :, :, :, :] = 1.0
        right_mask = base.reshape(1, n, 1).repeat(b, 1, c)
        left_mask = 1.0 - right_mask
        return ins * right_mask, ins * left_mask

    def _prepare_rgb_numpy(self):
        rgb_ctx = self._runtime_rgb
        if rgb_ctx is None:
            return None
        rgb = None
        if isinstance(rgb_ctx, (list, tuple)) and len(rgb_ctx) > 0:
            rgb = rgb_ctx[0]
        else:
            rgb = rgb_ctx
        if isinstance(rgb, (list, tuple)) and len(rgb) > 0:
            rgb = rgb[0]
        if torch.is_tensor(rgb):
            if rgb.dim() == 4:
                rgb = rgb[0]
            if rgb.dim() != 3:
                return None
            rgb = rgb.detach().float().cpu()
            if rgb.shape[0] == 3:
                rgb = rgb.permute(1, 2, 0)
            arr = rgb.numpy()
        else:
            arr = np.asarray(rgb)

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = arr * 255.0
            elif arr.min() < 0:
                arr = (arr + 1.0) * 127.5
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    @staticmethod
    def _to_seq_mask(mask2d: torch.Tensor, n: int, c: int, device, dtype):
        side = int(round(float(n) ** (1.0 / 3.0)))
        if side * side * side != n:
            flat = F.interpolate(mask2d[None, None], size=(n, 1), mode="nearest").reshape(1, n, 1)
            return flat.repeat(1, 1, c).to(device=device, dtype=dtype)
        resized = F.interpolate(mask2d[None, None], size=(side, side), mode="nearest")
        vol = resized.unsqueeze(2).repeat(1, 1, side, 1, 1)  # [1,1,D,H,W]
        seq = vol.reshape(1, n, 1).repeat(1, 1, c)
        return seq.to(device=device, dtype=dtype)

    def _fallback_split_masks(self, ins: torch.Tensor, image_h: int, image_w: int):
        left_2d = torch.zeros((image_h, image_w), dtype=torch.float32)
        right_2d = torch.zeros((image_h, image_w), dtype=torch.float32)
        mid = image_w // 2
        left_2d[:, :mid] = 1.0
        right_2d[:, mid:] = 1.0

        b, n, c = ins.shape
        right_seq = self._to_seq_mask(right_2d, n, c, ins.device, ins.dtype).repeat(b, 1, 1)
        left_seq = self._to_seq_mask(left_2d, n, c, ins.device, ins.dtype).repeat(b, 1, 1)
        return ins * right_seq, ins * left_seq

    def _try_grounded_sam(self, ins: torch.Tensor):
        if self._runtime_rgb is None:
            raise RuntimeError(
                "VisualTracker backend is grounded_sam but runtime RGB context is missing."
            )
        self._lazy_init_grounded_sam()
        if not self._gsam_ready:
            raise RuntimeError("VisualTracker: grounded_sam is not ready after initialization.")

        rgb_np = self._prepare_rgb_numpy()
        if rgb_np is None:
            raise RuntimeError("VisualTracker: failed to parse runtime RGB.")

        try:
            image_h, image_w = rgb_np.shape[:2]
            image_t, _ = self._gdino_transform(Image.fromarray(rgb_np), None)  # [3,H,W]
            prompt = self.prompt
            lang_text = ""
            if isinstance(self._runtime_lang, str):
                lang_text = self._runtime_lang.strip()
            elif isinstance(self._runtime_lang, (list, tuple)) and len(self._runtime_lang) > 0:
                lang_text = str(self._runtime_lang[0]).strip()
            elif self._runtime_lang is not None:
                lang_text = str(self._runtime_lang).strip()
            if lang_text:
                prompt = f"robotic arm. {lang_text}"
            boxes_filt, _ = self._get_grounding_output(
                self._gdino_model,
                image_t,
                prompt,
                self.box_threshold,
                self.text_threshold,
                device=self.device,
            )
            if boxes_filt is None or boxes_filt.numel() == 0:
                logging.warning(
                    "VisualTracker: grounded_sam returned no detection boxes; falling back to image-half masks."
                )
                return self._fallback_split_masks(ins, image_h, image_w)

            boxes = boxes_filt.clone()
            for i in range(boxes.shape[0]):
                boxes[i] = boxes[i] * torch.tensor([image_w, image_h, image_w, image_h], dtype=boxes.dtype)
                boxes[i][:2] -= boxes[i][2:] / 2
                boxes[i][2:] += boxes[i][:2]

            self._sam_predictor.set_image(rgb_np)
            transformed = self._sam_predictor.transform.apply_boxes_torch(
                boxes.cpu(), rgb_np.shape[:2]
            ).to(self.device)
            masks, _, _ = self._sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed,
                multimask_output=False,
            )
            if masks is None or masks.numel() == 0:
                logging.warning(
                    "VisualTracker: SAM returned empty masks; falling back to image-half masks."
                )
                return self._fallback_split_masks(ins, image_h, image_w)

            # Split detections by x-centroid as a robust no-label fallback.
            left_2d = torch.zeros((image_h, image_w), dtype=torch.float32)
            right_2d = torch.zeros((image_h, image_w), dtype=torch.float32)
            for i in range(masks.shape[0]):
                m = masks[i, 0].detach().cpu().float()
                x1, _, x2, _ = boxes[i].cpu().tolist()
                cx = 0.5 * (x1 + x2)
                if cx < image_w / 2.0:
                    left_2d = torch.maximum(left_2d, m)
                else:
                    right_2d = torch.maximum(right_2d, m)

            if left_2d.sum() < 1 and right_2d.sum() < 1:
                logging.warning(
                    "VisualTracker: both left/right masks are empty; falling back to image-half masks."
                )
                return self._fallback_split_masks(ins, image_h, image_w)
            if left_2d.sum() < 1:
                left_2d = 1.0 - right_2d
            if right_2d.sum() < 1:
                right_2d = 1.0 - left_2d

            b, n, c = ins.shape
            right_seq = self._to_seq_mask(right_2d, n, c, ins.device, ins.dtype)
            left_seq = self._to_seq_mask(left_2d, n, c, ins.device, ins.dtype)
            right_seq = right_seq.repeat(b, 1, 1)
            left_seq = left_seq.repeat(b, 1, 1)
            return ins * right_seq, ins * left_seq
        except Exception as exc:
            raise RuntimeError(f"VisualTracker: grounded_sam runtime failed ({exc})") from exc

    def forward(self, ins: torch.Tensor):
        if not self.enabled:
            raise RuntimeError("VisualTracker is disabled. Set framework.use_visual_tracker=True.")

        if self.backend != "grounded_sam":
            raise ValueError(f"Unsupported visual tracker backend: {self.backend}")

        out = self._try_grounded_sam(ins)
        if out is None:
            raise RuntimeError("VisualTracker: grounded_sam returned no valid mask output.")
        return out
