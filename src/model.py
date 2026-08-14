from __future__ import annotations

import math

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        alpha = 2.0
        surrogate = alpha / (1.0 + alpha * x.abs()).pow(2)
        return grad_output * surrogate


class LIFCell(nn.Module):
    def __init__(self, beta: float = 0.90, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(
        self, current: torch.Tensor, membrane: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if membrane is None:
            membrane = torch.zeros_like(current)
        membrane = self.beta * membrane + current
        spike = SurrogateSpike.apply(membrane - self.threshold)
        membrane = membrane * (1.0 - spike.detach())
        return spike, membrane


class TemporalChannelGate(nn.Module):

    def __init__(self, channels: int, initial_open_probability: float = 0.82):
        super().__init__()
        self.projection = nn.Linear(channels + 1, channels)
        nn.init.zeros_(self.projection.weight)
        initial_logit = math.log(initial_open_probability / (1.0 - initial_open_probability))
        nn.init.constant_(self.projection.bias, initial_logit)

    def forward(
        self, event_frame: torch.Tensor, previous_spikes: torch.Tensor | None
    ) -> torch.Tensor:
        event_density = event_frame.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        if previous_spikes is None:
            previous_rate = torch.zeros(
                event_frame.shape[0], self.projection.out_features,
                dtype=event_frame.dtype, device=event_frame.device,
            )
        else:
            previous_rate = previous_spikes.mean(dim=(2, 3))
        gate_features = torch.cat((event_density, previous_rate), dim=1)
        return torch.sigmoid(self.projection(gate_features))


class GatedConvSNN(nn.Module):
    def __init__(self, num_classes: int = 5, gate_enabled: bool = True):
        super().__init__()
        self.gate_enabled = gate_enabled

        self.stem = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 16),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(6, 24),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
        )
        self.lif1 = LIFCell()
        self.lif2 = LIFCell()
        self.lif3 = LIFCell()
        self.gate = TemporalChannelGate(16)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # frames: [batch, time, polarity, height, width]
        membrane1 = membrane2 = membrane3 = None
        previous_spikes = None
        logits_over_time = []
        gates = []
        spike_rates = []

        for time_index in range(frames.shape[1]):
            frame = frames[:, time_index]
            spikes1, membrane1 = self.lif1(self.stem(frame), membrane1)

            if self.gate_enabled:
                gate = self.gate(frame, previous_spikes)
            else:
                gate = torch.ones(
                    frame.shape[0], spikes1.shape[1], dtype=frame.dtype, device=frame.device
                )
            gated_spikes = spikes1 * gate[:, :, None, None]

            spikes2, membrane2 = self.lif2(self.block2(gated_spikes), membrane2)
            spikes3, membrane3 = self.lif3(self.block3(spikes2), membrane3)
            pooled = spikes3.mean(dim=(2, 3))
            logits_over_time.append(self.classifier(pooled))
            gates.append(gate)
            spike_rates.append(
                torch.stack((spikes1.mean(), spikes2.mean(), spikes3.mean()))
            )
            previous_spikes = spikes1

        logits = torch.stack(logits_over_time, dim=1).mean(dim=1)
        aux = {
            "gates": torch.stack(gates, dim=1),
            "spike_rates": torch.stack(spike_rates, dim=0),
        }
        return logits, aux
