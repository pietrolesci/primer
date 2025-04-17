import importlib.util

from lightning import seed_everything
from transformers import PreTrainedTokenizerFast  # type: ignore
from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


def get_model(name: str, tok: PreTrainedTokenizerFast) -> tuple[LlamaForCausalLM, PretrainedConfig]:
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

    if name == "me57M-tied":
        config = LlamaConfig(
            model_type="llama",
            hidden_act="silu",
            hidden_size=768,
            intermediate_size=3072,
            num_attention_heads=24,
            num_key_value_heads=24,
            num_hidden_layers=6,
            tie_word_embeddings=True,
            initializer_range=0.02,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=10000.0,
            **kwargs,
        )

    elif name in ("me100M", "me100M-tied"):
        # https://huggingface.co/HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints/blob/step-1200000/config.json
        config = LlamaConfig(
            model_type="llama",
            hidden_act="silu",
            hidden_size=576,
            intermediate_size=1536,
            num_attention_heads=9,
            num_key_value_heads=3,
            num_hidden_layers=30,
            tie_word_embeddings=name.endswith("-tied"),
            initializer_range=0.041666666666666664,
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=10000,  # note that during pre-training they use this
            rope_interleaved=False,
            **kwargs,
        )

    else:
        raise ValueError

    model = LlamaForCausalLM(config)

    return model, config
