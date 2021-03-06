import math

import torch
import torch.nn as nn

from representjs.models.encoder import CodeEncoder, CodeEncoderLSTM


class TypeTransformer(nn.Module):
    def __init__(
        self,
        n_tokens,
        n_output_tokens,
        d_model=512,
        d_rep=128,
        n_head=8,
        n_encoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        activation="relu",
        norm=True,
        pad_id=None,
        encoder_type="transformer"
    ):
        super(TypeTransformer, self).__init__()
        assert norm
        assert pad_id is not None
        self.config = {k: v for k, v in locals().items() if k != "self"}
        self.d_model = d_model
        self.n_tokens = n_tokens

        # Encoder and output for type prediction
        assert (encoder_type in ["transformer", "lstm"])
        if encoder_type == "transformer":
            self.encoder = CodeEncoder(
                n_tokens, d_model, d_rep, n_head, n_encoder_layers, d_ff, dropout, activation, norm, pad_id,
                project=False
            )
            # TODO: Try LeakyReLU
            # self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_output_tokens))
            self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.LayerNorm(d_model))

        elif encoder_type == "lstm":
            self.encoder = CodeEncoderLSTM(
                n_tokens=n_tokens,
                d_model=d_model,
                d_rep=d_rep,
                n_encoder_layers=n_encoder_layers,
                dropout=dropout,
                pad_id=pad_id,
                project=False
            )
            self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, n_output_tokens))

    def forward(self, src_tok_ids, lengths=None, output_attention=None):
        r"""
        Arguments:
            src_tok_ids: [B, L] long tensor
            output_attention: [B, L, L] float tensor
        """
        if output_attention is not None and src_tok_ids.size(0) != output_attention.size(0):
            raise RuntimeError("the batch number of src_tok_ids and output_attention must be equal")

        # Encode
        memory = self.encoder(src_tok_ids, lengths)  # LxBxD
        L, B, D = memory.shape
        memory = memory.transpose(0, 1)  # BxLxD

        if output_attention is not None:
            # Aggregate features to the starting token in each labeled identifier
            memory = torch.matmul(output_attention, memory)  # BxLxD

        memory = self.head(memory).view(L, B, self.d_model)  # L x B x D=d_model
        logits = torch.matmul(memory, self.encoder.embedding.weight.transpose(0, 1)).view(L, B,
                                                                                            self.n_tokens)  # [L, B, ntok]
        # return logits
        return torch.transpose(logits, 0, 1).view(B, L, self.n_tokens)  # [B, T, ntok]
        # # Predict logits over types
        # return self.head(memory)  # BxLxV
