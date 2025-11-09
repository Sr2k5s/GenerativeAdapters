# --- standard library ---
from collections.abc import Callable
from typing import Optional, Tuple, Union

# --- PyTorch ---
import torch
from torch import nn

# --- transformers: logging & common utils ---
from transformers.utils import logging, auto_docstring
from transformers.utils.generic import check_model_inputs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

# --- transformers: outputs, cache, generation, masking, attention utils ---
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_layers import GradientCheckpointingLayer

# --- transformers: activations & rope helpers ---
from transformers.activations import ACT2FN

# --- Gemma 2 (used as building blocks) ---
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2PreTrainedModel,
    Gemma2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

# --- Gemma 3 (pretrained base + rotary + scaled embedding + helpers) ---
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
    Gemma3CausalLMOutputWithPast,
    _bidirectional_window_overlay,
)

# --- adapter controllers (hyperformer) ---
from hyperformer.adapters import (
    AutoAdapterController,
    MetaAdapterConfig,
    TaskEmbeddingController,
    AdapterLayersHyperNetController,
    MetaLayersAdapterController,
    AdapterLayersOneHyperNetController,
)

# --- your local config for Gemma-3 text ---
from config_gemma3 import Gemma3TextConfig


logger = logging.get_logger(__name__)


class Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma2 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"



class Gemma2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Gemma2FFWrapper(nn.Module):
    def __init__(self, config, adapter_config = None):
        super().__init__()
        self.mlp_original_gemma = Gemma2MLP(config)
        self.train_adapters = getattr(config, "train_adapters", False)
        self.unique_hyper_net = False
        self.train_adapters_blocks = False
        if self.train_adapters and adapter_config is not None:
            self.unique_hyper_net = isinstance(adapter_config, MetaAdapterConfig) and (
                                                adapter_config.unique_hyper_net or adapter_config.efficient_unique_hyper_net
                                                )
            self.train_adapters_blocks = getattr(adapter_config, "train_adapters_blocks", False) and not self.unique_hyper_net
            if self.train_adapters_blocks:
                self.adapter_controller = AutoAdapterController.get(adapter_config)
                self.is_meta_adapter = isinstance(adapter_config, MetaAdapterConfig)
            elif self.unique_hyper_net:
                self.layer_hyper_net = MetaLayersAdapterController(adapter_config)

        self.layer_norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(getattr(config, "residual_dropout",
                                  getattr(config, "dropout_rate", 0.0)))


    def forward(self, hidden_states, task=None, task_embedding=None, gemma_adapters=None):
        norm_x = self.layer_norm(hidden_states)
        y = self.mlp_original_gemma(norm_x)
        if self.train_adapters and self.train_adapters_blocks:
            y = self.adapter_controller(task if not self.is_meta_adapter else task_embedding, y)
        elif self.train_adapters and self.unique_hyper_net:
            y = self.layer_hyper_net(y, gemma_adapters.feed_forward)
        layer_output = hidden_states + self.dropout(y)
        return layer_output


class Gemma2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        scale = float(getattr(config, "query_pre_attn_scalar", 1.0))
        self.scaling = (self.head_dim * scale) ** -0.5
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights




class Gemma2AttentionWrapper(nn.Module):
    def __init__(self, config, adapter_config=None, layer_idx : int = 0):
        super().__init__()
        self.attn = Gemma2Attention(config = config,
                                            layer_idx = layer_idx)
        
        self.train_adapters = getattr(config, "train_adapters", False)
        self.unique_hyper_net = False
        self.train_adapters_blocks = False
        if self.train_adapters and adapter_config is not None:
            self.unique_hyper_net = isinstance(adapter_config, MetaAdapterConfig) and (
                                                adapter_config.unique_hyper_net or adapter_config.efficient_unique_hyper_net
                                                )
            self.train_adapters_blocks = getattr(adapter_config, "train_adapters_blocks", False) and not self.unique_hyper_net
            if self.train_adapters_blocks:
                self.adapter_controller = AutoAdapterController.get(adapter_config)
                self.is_meta_adapter = isinstance(adapter_config, MetaAdapterConfig)
            elif self.unique_hyper_net:
                self.layer_hyper_net = MetaLayersAdapterController(adapter_config)

        self.rms_norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = nn.Dropout(getattr(config, "residual_dropout",
                                  getattr(config, "dropout_rate", 0.0)))


    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        task=None,
        task_embedding=None,
        gemma_adapters=None,          # expects .self_attention when unique_hyper_net=True
        **kwargs,                     # flash-attn kwargs etc.
    ):
        # Strip non-kernel kwargs; keep flash-attn knobs intact
        for k in ("position_ids", "use_cache", "task", "task_embedding", "gemma_adapters"):
            kwargs.pop(k, None)
        kwargs["output_attentions"] = bool(output_attentions)

        norm_x = self.rms_norm(hidden_states)
        attn_res = self.attn(
            hidden_states=norm_x,
            position_embeddings=position_embeddings,  # (cos, sin)
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        # Unpack robustly
        if isinstance(attn_res, tuple) and len(attn_res) == 2:
            attn_out, attn_weights = attn_res
        else:
            attn_out, attn_weights = attn_res, None

        y = attn_out
        if self.train_adapters:
            if self.train_adapters_blocks:
                y = self.adapter_controller(task if not getattr(self, "is_meta_adapter", False) else task_embedding, y)
            elif self.unique_hyper_net and gemma_adapters is not None:
                sa = getattr(gemma_adapters, "self_attention", None)
                if sa is not None:
                    y = self.layer_hyper_net(y, sa)

        layer_output = hidden_states + self.dropout(y)
        return (layer_output, attn_weights) if output_attentions else (layer_output,)




class Gemma2DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma2Config, layer_idx: int, adapter_config):
        super().__init__()
        self.adapter_config = adapter_config
        self.hidden_size = config.hidden_size
        self.config = config
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Gemma2AttentionWrapper(config=config, adapter_config = self.adapter_config, layer_idx=layer_idx)
        self.mlp = Gemma2FFWrapper(config=config, adapter_config = adapter_config)
        #self.input_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #self.post_attention_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        #self.pre_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #self.post_feedforward_layernorm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:

        # Self-attention (wrapper already does norm + residual)
        attn_tuple = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = attn_tuple[0]
        self_attn_weights = attn_tuple[1] if output_attentions and len(attn_tuple) > 1 else None

        # Feed-forward (wrapper already does norm + residual and adapter plumbing)
        hidden_states = self.mlp(
            hidden_states,
            task=kwargs.get("task"),
            task_embedding=kwargs.get("task_embedding"),
            gemma_adapters=kwargs.get("gemma_adapters"),
        )

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs





@auto_docstring
class Gemma2Model(Gemma2PreTrainedModel):
    def __init__(self, config: Gemma2Config, adapter_config=None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config, layer_idx,adapter_config=adapter_config) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )






from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.cache_utils import Cache, DynamicCache
import torch
from torch import nn
from typing import Optional, Union
from transformers.modeling_outputs import BaseModelOutputWithPast

class Gemma3TextModel(Gemma3PreTrainedModel):
    """
    Gemma-3 text-only decoder stack with adapter plumbing similar to T5Stack.

    Key parity with T5Stack:
      - accepts `adapter_config` in __init__
      - builds a ModuleList of decoder layers, each adapter-aware
      - supports unique/efficient hypernet controllers at the stack level
      - forwards `task`, `task_embedding`, and per-layer adapter weights (`gemma_adapters`)
      - final layer norm + (optional) residual dropout like T5Stack's final dropout
    """

    config: Gemma3TextConfig
    input_modalities = "text"

    def __init__(self, config: Gemma3TextConfig, embed_tokens: Optional[nn.Embedding] = None, adapter_config=None):
        super().__init__(config)
        self.adapter_config = adapter_config
        self.is_decoder = True  # Gemma text stack is a decoder-only transformer
        self.gradient_checkpointing = False


        # --- Embeddings ---
        # Gemma3 scales embeddings internally by sqrt(hidden_size)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = (
            embed_tokens
            if embed_tokens is not None
            else Gemma3TextScaledWordEmbedding(
                config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=config.hidden_size ** 0.5
            )
        )

        # --- Decoder layers (adapter-aware) ---
        # Like T5Stack(block=...), we create N layers that know about adapters.
        # Your decoder layer should accept `adapter_config` and wire it to attention/FF wrappers.
        self.layers = nn.ModuleList(
            [Gemma2DecoderLayer(config, layer_idx=i, adapter_config=self.adapter_config) for i in range(config.num_hidden_layers)]
        )

        # --- Adapter controllers at the stack level (T5 parity) ---
        self.train_adapters = bool(getattr(config, "train_adapters", False))
        self.unique_hyper_net = False
        self.efficient_unique_hyper_net = False
        if self.train_adapters and isinstance(adapter_config, MetaAdapterConfig):
            self.unique_hyper_net = bool(getattr(adapter_config, "unique_hyper_net", False))
            self.efficient_unique_hyper_net = bool(getattr(adapter_config, "efficient_unique_hyper_net", False))
            if self.unique_hyper_net:
                self.adapter_layers_hyper_net = AdapterLayersHyperNetController(adapter_config, config.num_hidden_layers)
            if self.efficient_unique_hyper_net:
                self.adapter_layers_hyper_net = AdapterLayersOneHyperNetController(adapter_config, config.num_hidden_layers)

        # --- Positional encoding & norms ---
        # Gemma3 rotary needs per-layer-type ROPE, so use the Gemma3RotaryEmbedding
        self.rotary_emb = Gemma3RotaryEmbedding(config)
        self.final_layer_norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # T5Stack uses `config.dropout_rate`; Gemma configs typically don't define it.
        # Use residual_dropout if provided, otherwise fall back to attention_dropout (often 0.0).
        p = float(getattr(config, "residual_dropout", getattr(config, "attention_dropout", 0.0)))
        self.dropout = nn.Dropout(p)

        # Initialize weights and apply final processing
        self.post_init()

    # --- T5Stack parity helpers ---
    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        # Decoder-only; returning embeddings mirrors T5Stack interface
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embed_tokens = new_embeddings

    # --- Forward ---
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,

        # --- adapter hooks (T5 parity) ---
        task: Optional[str] = None,
        task_embedding: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:

        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        # Exactly one of input_ids / inputs_embeds
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")

        if self.training and self.gradient_checkpointing and use_cache:
            # Same behavior as HF stacks
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Create/advance cache positions
        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # --- Attention masks (Gemma3 has per-layer-type masks) ---
        # Build mapping for {"full_attention": mask, "sliding_attention": mask}
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_kwargs = dict(mask_kwargs)
            if getattr(self.config, "use_bidirectional_attention", False):
                sliding_kwargs["or_mask_function"] = _bidirectional_window_overlay(self.config.sliding_window)
            sliding_mask = create_sliding_window_causal_mask(**sliding_kwargs)

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": sliding_mask,
            }

        else:
            causal_mask_mapping = attention_mask  # already prepared

        # --- Rotary embeddings per layer type (Gemma3 style) ---
        # Precompute (cos, sin) for each attention type present in config.layer_types
        position_embeddings = {}
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(inputs_embeds, position_ids, layer_type=layer_type)

        # --- Adapter hyper-net (per-layer) like T5Stack ---
        # If using unique/efficient hypernets, compute layer-specific adapter weights container
        

        # --- Run decoder layers ---
        hidden_states = self.dropout(inputs_embeds)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            
            gemma_adapters = None
            if self.train_adapters and (self.unique_hyper_net or self.efficient_unique_hyper_net):
                gemma_adapters = self.adapter_layers_hyper_net(task_embedding, i)


            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,

                # adapter signals
                task=task,
                task_embedding=task_embedding,
                gemma_adapters=gemma_adapters,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)

        # --- Final norm + (optional) dropout ---
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # T5Stack returns (hidden, past, hidden_states, attn, cross_attn)
            # Here: no cross-attn in text-only stack, and past is carried via `past_key_values`
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )





class Gemma3ForConditionalGeneration(Gemma3PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]
    accepts_loss_kwargs = False

    def __init__(self, config, adapter_config=None):
        super().__init__(config)
        self.adapter_config = adapter_config
        self.train_adapters = bool(getattr(config, "train_adapters", False))

        if self.train_adapters and isinstance(adapter_config, MetaAdapterConfig):
            self.task_embedding_controller = TaskEmbeddingController(adapter_config)
        else:
            self.task_embedding_controller = None

        # Choose the right text config
        self.text_config = getattr(config, "text_config", config)

        # Text-only stack
        self.model = Gemma3TextModel(self.text_config, adapter_config=adapter_config)

        # LM head matches text dims
        self.lm_head = nn.Linear(self.text_config.hidden_size, self.text_config.vocab_size, bias=False)

        self.post_init()

    # tie-weights support
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,  # accepted but unused in text-only
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        token_type_ids: Optional[torch.LongTensor] = None,  # accepted but unused
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        task: Optional[str] = None,
        task_embedding: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ) -> Union[tuple, Gemma3CausalLMOutputWithPast]:

        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        if (
            task_embedding is None
            and self.train_adapters
            and isinstance(self.adapter_config, MetaAdapterConfig)
            and self.task_embedding_controller is not None
        ):
            task_embedding = self.task_embedding_controller(task)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            task=task,
            task_embedding=task_embedding,
            **lm_kwargs,
        )

        hidden_states = outputs[0]
        # logits slice
        if isinstance(logits_to_keep, int):
            # -0 slices to 0 in Python; use full sequence when 0
            sl = slice(None) if logits_to_keep == 0 else slice(-logits_to_keep, None)
        else:
            sl = logits_to_keep

        logits = self.lm_head(hidden_states[:, sl, :])

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask != 0].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            flat_logits = shift_logits.view(-1, self.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
