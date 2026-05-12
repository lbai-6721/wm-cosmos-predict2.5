from cosmos_predict2._src.predict2.action.configs.action_conditioned.config import Config
from cosmos_predict2._src.predict2.action.configs.action_conditioned.config import make_config as _make_base_config

from wm4vla.configs.tokenizer_registry import register_wm4vla_tokenizers


def make_config() -> Config:
    config = _make_base_config()
    register_wm4vla_tokenizers()
    return config
