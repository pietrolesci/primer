import importlib.util

from lightning import seed_everything
from transformers import PreTrainedTokenizerFast  # type: ignore
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from primer.utilities import add_rich_handler, get_logger

logger = add_rich_handler(get_logger("model"))


def get_model(model_config: dict, tok: PreTrainedTokenizerFast) -> tuple[LlamaForCausalLM, PretrainedConfig]:
    # Dynamically check if flash attention is available using importlib
    attn_implementation = "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"

    kwargs = {
        "vocab_size": tok.vocab_size,
        "bos_token_id": tok.bos_token_id,  # type: ignore
        "eos_token_id": tok.eos_token_id,  # type: ignore
        "pad_token_id": tok.pad_token_id,  # type: ignore
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "max_position_embeddings": 2048,
        "_attn_implementation": attn_implementation,
    }
    seed_everything(42)
    model_config = {**kwargs, **model_config}
    config = LlamaConfig(**model_config)
    model = LlamaForCausalLM(config)
    logger.info(
        f"Model config:\n{model.config.to_json_string()}\n"
        f"Attention implementation: {model.config._attn_implementation}\n"
        f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f}MB\n"
        f"Num parameters: {model.num_parameters() / 1e6:.1f}M"
    )

    return model, config
