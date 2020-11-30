from examples.brainspeech.models.vgglas import DEFAULT_ENC_VGGBLOCK_CONFIG, DEFAULT_MAX_TARGET_POSITIONS
from examples.brainspeech.models.vgglas.encoder import BrainLSTMEncoder
from examples.brainspeech.models.vgglas.decoder import BrainLSTMDecoder
from torch import Tensor
from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoderDecoderModel
)
from fairseq import options
from typing import Optional, Dict


@register_model("fairseq_las")
class FairseqListenAttendSpell(FairseqEncoderDecoderModel):
    """
    VGG + Listen, Attend and Spell Architecture.


    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        # Listener
        parser.add_argument('--encoder-input-dim', type=int)
        parser.add_argument('--vggblock-config', type=tuple)
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N', help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', '-encoder-bidirectional',
                            action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')

        # Speller
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N', help='decoder embedding dimension')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N', help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N', help='decoder output embedding dimension')
        parser.add_argument('--decoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for decoder output')
        parser.add_argument('--share-decoder-input-output-embed', default=False,
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        max_target_positions = getattr(args, 'max_target_positions', DEFAULT_MAX_TARGET_POSITIONS)

        encoder = BrainLSTMEncoder(
            dictionary=task.target_dict,
            input_dim=args.encoder_input_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional
        )
        decoder = BrainLSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=True,
            encoder_output_units=encoder.output_units,
            pretrained_embed=None,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
            max_target_positions=max_target_positions,
            residuals=False
        )
        return cls(encoder, decoder)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state)

        return decoder_out


@register_model_architecture('brain_las', 'brain_las_1')
def base_architecture(args):
    args.encoder_input_dim = getattr(args, 'encoder_input_dim', 80)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.2)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.2)
    args.vggblock_config = getattr(args, 'vggblock_config', DEFAULT_ENC_VGGBLOCK_CONFIG)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.2)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.2)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')


@register_model_architecture('brain_las', 'brain_las_2')
def base_architecture(args):
    args.encoder_input_dim = getattr(args, 'encoder_input_dim', 80)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.2)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.2)
    args.vggblock_config = getattr(args, 'vggblock_config', DEFAULT_ENC_VGGBLOCK_CONFIG)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 1024)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_attention = getattr(args, 'decoder_attention', '1')
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.2)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.2)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '10000,50000,200000')