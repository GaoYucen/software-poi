from __future__ import annotations

from transformers import GPT2Config, GPT2Model
import torch
from torch import nn


class GPTBackbone(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        model_name: str,
        use_pretrained: bool,
        freeze: bool,
        unfreeze_last_n: int,
        layers: int,
        heads: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if use_pretrained:
            try:
                self.gpt = GPT2Model.from_pretrained(model_name)
                source_dim = self.gpt.config.n_embd
            except Exception as exc:
                print(f"Could not load pretrained GPT-2 '{model_name}': {exc}")
                print("Falling back to randomly initialized GPT-2 architecture.")
                config = GPT2Config(
                    n_embd=hidden_dim,
                    n_layer=layers,
                    n_head=heads,
                    n_positions=max_seq_len,
                    n_ctx=max_seq_len,
                    resid_pdrop=dropout,
                    embd_pdrop=dropout,
                    attn_pdrop=dropout,
                )
                self.gpt = GPT2Model(config)
                source_dim = hidden_dim
        else:
            config = GPT2Config(
                n_embd=hidden_dim,
                n_layer=layers,
                n_head=heads,
                n_positions=max_seq_len,
                n_ctx=max_seq_len,
                resid_pdrop=dropout,
                embd_pdrop=dropout,
                attn_pdrop=dropout,
            )
            self.gpt = GPT2Model(config)
            source_dim = hidden_dim
        self.input_projection = nn.Identity() if hidden_dim == source_dim else nn.Linear(hidden_dim, source_dim)
        self.output_projection = nn.Identity() if hidden_dim == source_dim else nn.Linear(source_dim, hidden_dim)
        if freeze:
            for parameter in self.gpt.parameters():
                parameter.requires_grad = False
            if unfreeze_last_n > 0:
                for block in self.gpt.h[-unfreeze_last_n:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                for parameter in self.gpt.ln_f.parameters():
                    parameter.requires_grad = True

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(inputs_embeds)
        output = self.gpt(inputs_embeds=projected, attention_mask=attention_mask, use_cache=False)
        return self.output_projection(output.last_hidden_state)


def last_valid_state(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_idx, lengths]
