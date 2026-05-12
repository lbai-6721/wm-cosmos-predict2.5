from __future__ import annotations

from typing import Optional, Sequence

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference as BaseVideo2WorldInference
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint

from wm4vla.tokenizers.wan2pt1_lightvae import DEFAULT_LIGHTVAE_PTH


def resolve_tokenizer_overrides(
    tokenizer_backend: str = "wan2pt1",
    tokenizer_vae_pth: str = "",
    lightx2v_root: str = "",
    use_lightvae: bool = False,
    lightvae_pth: str = DEFAULT_LIGHTVAE_PTH,
    experiment_opts: Optional[Sequence[str]] = None,
) -> tuple[str, str, list[str]]:
    backend = "lightvae" if use_lightvae else tokenizer_backend
    vae_pth = tokenizer_vae_pth or (lightvae_pth if backend == "lightvae" else "")
    opts = list(experiment_opts or [])
    if backend == "lightvae":
        opts.extend(
            [
                "tokenizer=wan2pt1_lightvae_tokenizer",
                f"model.config.tokenizer.vae_pth={vae_pth}",
            ]
        )
        if lightx2v_root:
            opts.append(f"model.config.tokenizer.lightx2v_root={lightx2v_root}")
    elif tokenizer_vae_pth:
        opts.extend(
            [
                "tokenizer=wan2pt1_tokenizer",
                f"model.config.tokenizer.vae_pth={tokenizer_vae_pth}",
            ]
        )
    return backend, vae_pth, opts


class Video2WorldInference(BaseVideo2WorldInference):
    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str,
        context_parallel_size: int = 1,
        config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py",
        experiment_opts: Optional[Sequence[str]] = None,
    ):
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None

        if self.context_parallel_size > 1:
            self._init_distributed()

        merged_experiment_opts = list(experiment_opts or [])
        if not INTERNAL:
            merged_experiment_opts.append("~data_train")

        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=config_file,
            load_ema_to_reg=True,
            experiment_opts=merged_experiment_opts,
        )

        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)

        self.model = model
        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None
