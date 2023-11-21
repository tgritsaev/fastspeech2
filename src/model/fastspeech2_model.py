import torch
from torch import nn

from src.model.base_model import BaseModel
from src.model.transformer import Encoder, Decoder
from src.model.variance_adaptor import VarianceAdaptor


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech2Model(BaseModel):
    """FastSpeech"""

    def __init__(
        self,
        max_seq_len,
        vocab_size,
        fft_conv1d_kernel,
        fft_conv1d_padding,
        encoder_n_layer,
        encoder_dim,
        encoder_head,
        encoder_conv1d_filter_size,
        duration_predictor_filter_size,
        duration_predictor_kernel_size,
        pitch_predictor_filter_size,
        pitch_predictor_kernel_size,
        energy_predictor_filter_size,
        energy_predictor_kernel_size,
        decoder_n_layer,
        decoder_dim,
        decoder_head,
        decoder_conv1d_filter_size,
        dropout,
        PAD,
        min_pitch,
        max_pitch,
        min_energy,
        max_energy,
        num_bins,
        num_mels,
    ):
        super().__init__()

        self.encoder = Encoder(
            max_seq_len,
            encoder_n_layer,
            vocab_size,
            encoder_dim,
            encoder_conv1d_filter_size,
            encoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            PAD,
            dropout,
        )

        self.variance_adaptor = VarianceAdaptor(
            encoder_dim,
            duration_predictor_filter_size,
            duration_predictor_kernel_size,
            pitch_predictor_filter_size,
            pitch_predictor_kernel_size,
            energy_predictor_filter_size,
            energy_predictor_kernel_size,
            min_pitch,
            max_pitch,
            min_energy,
            max_energy,
            num_bins,
            dropout,
        )

        self.decoder = Decoder(
            max_seq_len,
            decoder_n_layer,
            decoder_dim,
            decoder_conv1d_filter_size,
            decoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            PAD,
            dropout,
        )
        self.mel_linear = nn.Linear(decoder_dim, num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.0)

    def forward(
        self,
        src_seq,
        src_pos,
        pitch_target=None,
        energy_target=None,
        mel_pos=None,
        mel_max_length=None,
        length_target=None,
        length_control=1,
        pitch_control=1,
        energy_control=1,
        **kwargs
    ):
        enc_output, _ = self.encoder(src_seq, src_pos)
        output, length_prediction, pitch_prediction, energy_prediction = self.variance_adaptor(
            enc_output, length_target, pitch_target, energy_target, mel_max_length, length_control, pitch_control, energy_control
        )
        if self.training:
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            return {
                "mel_prediction": self.mel_linear(output),
                "length_prediction": length_prediction,
                "pitch_prediction": pitch_prediction,
                "energy_prediction": energy_prediction,
            }
        else:
            output = self.decoder(output, length_prediction)
            return {"mel_prediction": self.mel_linear(output)}
