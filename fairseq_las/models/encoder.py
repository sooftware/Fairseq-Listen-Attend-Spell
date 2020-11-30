import torch.nn as nn
from torch import Tensor
from collections.abc import Iterable
from typing import Optional
from fairseq_las.models import DEFAULT_ENC_VGGBLOCK_CONFIG
from fairseq_las.models.modules import lstm
from fairseq_las.models.vggblock import VGGBlock
from fairseq.models import FairseqEncoder
from fairseq.modules import FairseqDropout


class FairseqListener(FairseqEncoder):
    def __init__(
            self,
            dictionary,
            input_dim: int = 80,
            vggblock_config: tuple = DEFAULT_ENC_VGGBLOCK_CONFIG,
            in_channels: int = 1,
            hidden_size: int = 512,
            num_layers: int = 3,
            dropout_in: float = 0.1,
            dropout_out: float = 0.1,
            bidirectional: bool = True
    ):
        super().__init__(dictionary)
        self.num_vgg_blocks = 0
        self.padding_idx = dictionary.pad()
        self.dropout_in_module = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)

        if vggblock_config is not None:
            if not isinstance(vggblock_config, Iterable):
                raise ValueError("vggblock_config is not iterable")
            self.num_vggblocks = len(vggblock_config)

        self.conv_layers = nn.ModuleList()
        self.in_channels = in_channels
        self.input_dim = input_dim
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if vggblock_config is not None:
            for _, config in enumerate(vggblock_config):
                (
                    out_channels,
                    conv_kernel_size,
                    pooling_kernel_size,
                    num_conv_layers,
                    layer_norm
                ) = config
                self.conv_layers.append(
                    VGGBlock(
                        in_channels,
                        out_channels,
                        conv_kernel_size,
                        pooling_kernel_size,
                        num_conv_layers,
                        input_dim=input_dim,
                        layer_norm=layer_norm
                    )
                )
                in_channels = out_channels
                input_dim = self.conv_layers[-1].output_dim

        self.lstm = lstm(
            input_size=input_dim << 7,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out_module.p if num_layers > 1 else 0.,
            bidirectional=bidirectional
        )
        self.output_units = hidden_size
        if bidirectional:
            self.output_units <<= 1

    def forward(self, src_tokens: Tensor, src_lengths: Tensor = Optional[None], **kwargs):
        """
        src_tokens: padded tensor B x T x C * feat
        src_lengths: tensor of original lengths of input utterances B
        """
        batch_size = src_tokens.size(0)
        seq_length = src_tokens.size(1)

        x = src_tokens.view(batch_size, seq_length, self.in_channels, self.input_dim)
        x = x.transpose(1, 2).contiguous()

        for idx in range(len(self.conv_layers)):
            x = self.conv_layers[idx](x)

        batch_size = x.size(0)
        output_length = x.size(2)

        # (B x C x T x feat) => (B x T x C x feat) => (T x B x C x feat) => (T x B x C * feat)
        x = x.transpose(1, 2).transpose(0, 1)
        x = x.contiguous().view(output_length, batch_size, -1)

        subsampling_factor = int(seq_length * 1.0 / output_length + 0.5)
        input_lengths = (src_lengths.float() / subsampling_factor).ceil().long()

        packed_x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, enforce_sorted=True)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:
            state_size = self.num_layers, batch_size, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx * 1.0)
        x = self.dropout_out_module(x)

        if self.bidirectional:
            final_hiddens = self.combine_bidirection(final_hiddens, batch_size)
            final_cells = self.combine_bidirection(final_cells, batch_size)

        return tuple((
            x,
            final_hiddens,
            final_cells
        ))

    def combine_bidirection(self, outs, batch_size: int):
        out = outs.view(self.num_layers, 2, batch_size, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, batch_size, -1)

    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple((
            encoder_out[0].index_select(1, new_order),
            encoder_out[1].index_select(1, new_order),
            encoder_out[2].index_select(1, new_order),
            None
        ))
