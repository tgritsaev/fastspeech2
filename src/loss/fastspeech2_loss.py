import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        mel_prediction,
        length_prediction,
        energy_prediction,
        pitch_prediction,
        mel_target,
        length_target,
        energy_target,
        pitch_target,
        **kwargs
    ):
        mel_loss = self.l1_loss(mel_prediction, mel_target)
        duration_loss = self.mse_loss(length_prediction, torch.log1p(length_target.float()))
        energy_loss = self.mse_loss(energy_prediction, torch.log1p(energy_target))
        pitch_loss = self.mse_loss(pitch_prediction, torch.log1p(pitch_target))

        return {
            "loss": mel_loss + duration_loss + energy_loss + pitch_loss,
            "mel_loss": mel_loss,
            "duration_loss": duration_loss,
            "energy_loss": energy_loss,
            "pitch_loss": pitch_loss,
        }
