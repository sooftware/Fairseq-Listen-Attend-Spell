import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from fairseq_las.models import DEFAULT_MAX_TARGET_POSITIONS
from fairseq_las.models.modules import linear, embedding, lstm_cell
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.lstm import AttentionLayer
from fairseq.modules import FairseqDropout, AdaptiveSoftmax


class FairseqSpeller(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        residuals=False,
    ):
        super().__init__(dictionary)
        self.dropout_in_module = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList([
            lstm_cell(input_size=input_feed_size + embed_dim if layer == 0 else hidden_size, hidden_size=hidden_size)
            for layer in range(num_layers)
        ])

        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=True)
        else:
            self.attention = None

        if hidden_size != out_embed_dim:
            self.additional_fc = linear(hidden_size, out_embed_dim)

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings, hidden_size, adaptive_softmax_cutoff, dropout=dropout_out
            )
        elif not self.share_input_output_embed:
            self.fc_out = linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        src_lengths: Optional[Tensor] = None,
    ):
        x, attn_scores = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
        return self.output_layer(x), attn_scores

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        """
        Similar to *forward* but only return features.
        """
        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs = encoder_out[0]
            encoder_hidden = encoder_out[1]
            encoder_cells = encoder_out[2]
        else:
            encoder_outs = torch.empty(0)
            encoder_hidden = torch.empty(0)
            encoder_cells = torch.empty(0)
        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hidden, prev_cells, input_feed = self.get_cached_state(incremental_state)
        elif encoder_out is not None:
            # setup recurrent cells
            prev_hidden = [encoder_hidden[i] for i in range(self.num_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hidden = [self.encoder_hidden_proj(y) for y in prev_hidden]
                prev_cells = [self.encoder_cell_proj(y) for y in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
        else:
            # setup zero cells, since there is no encoder
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hidden = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = None

        assert srclen > 0 or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input_var = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input_var = x[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input_var, (prev_hidden[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input_var = self.dropout_out_module(hidden)
                if self.residuals:
                    input_var += prev_hidden[i]

                # save state for next time step
                prev_hidden[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                assert attn_scores is not None
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask=None)
            else:
                out = hidden
            out = self.dropout_out_module(out)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final output
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hidden_tensor = torch.stack(prev_hidden)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hidden": prev_hidden_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            }
        )
        self.set_incremental_state(incremental_state, 'cached_state', cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def get_cached_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        cached_state = self.get_incremental_state(incremental_state, 'cached_state')
        assert cached_state is not None
        prev_hidden_ = cached_state["prev_hidden"]
        assert prev_hidden_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hidden = [prev_hidden_[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        input_feed = cached_state["input_feed"]  # can be None for decoder-only language models
        return prev_hidden, prev_cells, input_feed

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hidden, prev_cells, input_feed = self.get_cached_state(incremental_state)
        prev_hidden = [p.index_select(0, new_order) for p in prev_hidden]
        prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hidden": torch.stack(prev_hidden),
                "prev_cells": torch.stack(prev_cells),
                "input_feed": input_feed,
            }
        )
        self.set_incremental_state(incremental_state, 'cached_state', cached_state_new),
        return

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn
