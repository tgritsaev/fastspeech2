import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def create_alignment(base_mat, duration_prediction):
    N, L = duration_prediction.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_prediction[i][j]):
                base_mat[i][count + k][j] = 1
            count = count + duration_prediction[i][j]
    return base_mat


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class Predictor(nn.Module):
    """Duration Predictor"""

    def __init__(self, encoder_dim, predictor_filter_size, predictor_kernel_size, dropout):
        super().__init__()

        self.input_size = encoder_dim
        self.filter_size = predictor_filter_size
        self.kernel = predictor_kernel_size
        self.conv_output_size = predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(self.input_size, self.filter_size, kernel_size=self.kernel, padding=1),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self, encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super().__init__()
        self.duration_predictor = Predictor(encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)

    def LR(self, x, duration_prediction, mel_max_length=None):
        expand_max_len = torch.max(torch.sum(duration_prediction, -1), -1)[0]
        alignment = torch.zeros(duration_prediction.size(0), expand_max_len, duration_prediction.size(1)).numpy()
        alignment = create_alignment(alignment, duration_prediction.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(output, (0, 0, 0, mel_max_length - output.size(1), 0, 0))
        return output

    def forward(self, x, control=1.0, target=None, mel_max_length=None):
        duration_prediction = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_prediction
        else:
            duration_prediction = (((torch.exp(duration_prediction) - 1) * control) + 0.5).int()
            duration_prediction[duration_prediction < 0] = 0

            output = self.LR(x, duration_prediction)
            mel_pos = torch.stack([torch.arange(1, output.size(1) + 1, device=x.device)]).long()

            return output, mel_pos


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
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
    ):
        super().__init__()
        self.length_regulator = LengthRegulator(encoder_dim, duration_predictor_filter_size, duration_predictor_kernel_size, dropout)

        self.pitch_predictor = Predictor(encoder_dim, pitch_predictor_filter_size, pitch_predictor_kernel_size, dropout)
        pitch_bins = torch.linspace(np.log(min_pitch + 1), np.log(max_pitch + 2), num_bins)
        self.register_buffer("pitch_bins", pitch_bins)
        self.pitch_embedding = nn.Embedding(num_bins, encoder_dim)

        # we estimane energy_target + 1, so we add +1 to bounds
        self.energy_predictor = Predictor(encoder_dim, energy_predictor_filter_size, energy_predictor_kernel_size, dropout)
        energy_bins = torch.linspace(np.log(min_energy + 1), np.log(max_energy + 2), num_bins)
        self.register_buffer("energy_bins", energy_bins)
        self.energy_embedding = nn.Embedding(num_bins, encoder_dim)

    def get(self, x, param: str, target=None, control=1.0):
        if param == "energy":
            prediction = self.energy_predictor(x)
            bins = self.energy_bins
        else:
            prediction = self.pitch_predictor(x)
            bins = self.pitch_bins

        if target is not None:
            buckets = torch.bucketize(torch.log(target + 1), bins)
        else:
            estimated = (torch.exp(prediction) - 1) * control
            buckets = torch.bucketize(torch.log(estimated + 1), bins)

        if param == "energy":
            embedding = self.energy_embedding(buckets)
        else:
            embedding = self.pitch_embedding(buckets)
        return embedding, prediction

    def forward(
        self, enc_output, length_target, pitch_target, energy_target, mel_max_length, length_control, pitch_control, energy_control
    ):
        len_reg_output, length_prediction = self.length_regulator(enc_output, length_control, length_target, mel_max_length)

        pitch_embedding, pitch_prediction = self.get(len_reg_output, "pitch", pitch_target, pitch_control)
        energy_embedding, energy_prediction = self.get(len_reg_output, "energy", energy_target, energy_control)

        output = len_reg_output + pitch_embedding + energy_embedding
        return output, length_prediction, pitch_prediction, energy_prediction
