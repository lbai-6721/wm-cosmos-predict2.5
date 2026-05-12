from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict

from wm4vla.tokenizers.wan2pt1_lightvae import Wan2pt1LightVAEInterface

Wan2pt1LightVAEConfig: LazyDict = L(Wan2pt1LightVAEInterface)(
    name="wan2pt1_lightvae_tokenizer",
    vae_pth=None,
    lightx2v_root=None,
    use_batched_vae=True,
)


def register_wm4vla_tokenizers():
    cs = ConfigStore.instance()
    cs.store(
        group="tokenizer",
        package="model.config.tokenizer",
        name="wan2pt1_lightvae_tokenizer",
        node=Wan2pt1LightVAEConfig,
    )
