from __future__ import annotations

import importlib
import pathlib
import sys
import types
from typing import Optional

import torch
import torch.distributed as distributed

from cosmos_predict2._src.predict2.tokenizers.interface import VideoTokenizerInterface

DEFAULT_LIGHTX2V_ROOT = "/home/kyji/storage_net/tmp/lbai/LightX2V"
DEFAULT_LIGHTVAE_PTH = "/home/kyji/public/models/lightx2v/vae/lightvaew2_1.pth"


def _import_lightx2v_wan_vae(lightx2v_root: Optional[str] = None):
    try:
        from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightX2VWanVAE

        return LightX2VWanVAE
    except ModuleNotFoundError as exc:
        root = pathlib.Path(lightx2v_root or DEFAULT_LIGHTX2V_ROOT)
        if root.exists():
            package_dir = root / "lightx2v"
            if not package_dir.exists():
                raise ModuleNotFoundError(
                    f"Invalid lightx2v_root: {root}. Expected directory: {package_dir}"
                ) from exc

            if str(root) not in sys.path:
                sys.path.insert(0, str(root))

            # Avoid importing the heavy top-level package when only WanVAE is needed.
            sys.modules.pop("lightx2v", None)
            for name in list(sys.modules.keys()):
                if name.startswith("lightx2v."):
                    sys.modules.pop(name, None)
            ns_pkg = types.ModuleType("lightx2v")
            ns_pkg.__path__ = [str(package_dir)]
            ns_pkg.__file__ = str(package_dir / "__init__.py")
            sys.modules["lightx2v"] = ns_pkg

            from lightx2v_platform.base import global_var as lightx2v_global_var
            from lightx2v_platform.base.base import check_ai_device, init_ai_device

            if not isinstance(lightx2v_global_var.AI_DEVICE, str) or not lightx2v_global_var.AI_DEVICE:
                init_ai_device("cuda")
                check_ai_device("cuda")

            module = importlib.import_module("lightx2v.models.video_encoders.hf.wan.vae")
            return module.WanVAE
        raise ModuleNotFoundError(
            "Failed to import lightx2v. Set PYTHONPATH to include the LightX2V repo root "
            "or pass `lightx2v_root` in tokenizer config."
        ) from exc


class Wan2pt1LightVAEInterface(VideoTokenizerInterface):
    def __init__(self, chunk_duration: int = 81, load_mean_std: bool = False, **kwargs):
        del load_mean_std
        self.keep_decoder_cache = kwargs.get("keep_decoder_cache", False)
        self.keep_encoder_cache = kwargs.get("keep_encoder_cache", False)
        self.cp_initialized = False
        self.chunk_duration = chunk_duration
        self.use_batched_vae = kwargs.get("use_batched_vae", True)
        self._warned_parallel_fallback = False

        lightx2v_root = kwargs.get("lightx2v_root") or DEFAULT_LIGHTX2V_ROOT
        LightX2VWanVAE = _import_lightx2v_wan_vae(lightx2v_root)

        self.model = LightX2VWanVAE(
            dtype=torch.bfloat16,
            device=kwargs.get("device", "cuda"),
            vae_path=kwargs.get("vae_pth") or DEFAULT_LIGHTVAE_PTH,
            parallel=kwargs.get("is_parallel", False),
            use_2d_split=kwargs.get("use_2d_split", True),
            load_from_rank0=kwargs.get("load_from_rank0", False),
            use_lightvae=True,
        )

    def _can_use_batched_vae(self) -> bool:
        if not self.use_batched_vae:
            return False
        if getattr(self.model, "parallel", False):
            if not self._warned_parallel_fallback:
                print(
                    "[warn] use_batched_vae=True but LightX2V parallel=True. "
                    "Falling back to legacy per-sample path."
                )
                self._warned_parallel_fallback = True
            return False
        return True

    @staticmethod
    def _ensure_5d(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == 5:
            return x
        if x.ndim == 4:
            return x.unsqueeze(0)
        raise ValueError(f"Unexpected {name} shape: {x.shape}")

    def initialize_context_parallel(self, cp_group: distributed.ProcessGroup, cp_grid_shape: tuple[int, int]) -> None:
        assert self.cp_initialized is False
        if hasattr(self.model, "_initialize_context_parallel"):
            self.cp_initialized = True
            self.model._initialize_context_parallel(cp_group, cp_grid_shape)

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    def clear_cache(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "clear_cache"):
            self.model.model.clear_cache()

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        input_dtype = state.dtype
        state = state.to(dtype=self.dtype)
        if state.ndim == 5 and self._can_use_batched_vae():
            latent = self.model.model.encode(state, self.model.scale)
            latent = self._ensure_5d(latent, "latent")
            return latent.to(input_dtype)
        if state.ndim == 5 and state.shape[0] > 1:
            latent = torch.stack(
                [self.model.encode(state[i : i + 1], world_size_h=None, world_size_w=None) for i in range(state.shape[0])],
                dim=0,
            )
        else:
            latent = self.model.encode(state, world_size_h=None, world_size_w=None)
        return latent.to(input_dtype)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        input_dtype = latent.dtype
        latent = latent.to(dtype=self.dtype).contiguous()
        if latent.ndim == 5 and self._can_use_batched_vae():
            recon = self.model.model.decode(latent, self.model.scale).clamp_(-1, 1)
            recon = self._ensure_5d(recon, "reconstruction")
            return recon.to(input_dtype)
        if latent.ndim == 5 and latent.shape[0] > 1:
            recon_list = []
            for i in range(latent.shape[0]):
                recon_i = self.model.decode(latent[i].contiguous())
                if isinstance(recon_i, list):
                    assert len(recon_i) == 1, "Assuming batch_size=1 was used"
                    recon_i = recon_i[0].unsqueeze(0)
                if recon_i.ndim == 4:
                    recon_i = recon_i.unsqueeze(0)
                recon_list.append(recon_i)
            recon = torch.cat(recon_list, dim=0)
        else:
            recon = self.model.decode(latent)
            if isinstance(recon, list):
                assert len(recon) == 1, "Assuming batch_size=1 was used"
                recon = recon[0].unsqueeze(0)
        return recon.to(input_dtype)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return 1 + (num_pixel_frames - 1) // 4

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (num_latent_frames - 1) * 4 + 1

    @property
    def spatial_compression_factor(self):
        return 8

    @property
    def temporal_compression_factor(self):
        return 4

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self):
        return 16

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "wan2pt1_lightvae_tokenizer"
