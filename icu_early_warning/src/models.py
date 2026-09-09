import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_features + 1, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        # Give the baseline the elapsed-time feature too; this makes the comparison fair.
        z = torch.cat([x, delta_t.unsqueeze(-1)], dim=-1)
        out, _ = self.rnn(z)
        return self.head(out[:, -1]).squeeze(-1)


class CfCClassifier(nn.Module):
    """Liquid/CfC model. ncps consumes a per-step elapsed-time tensor."""
    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        try:
            from ncps.torch import CfC
        except ImportError as exc:
            raise ImportError("Install ncps: pip install ncps") from exc
        self.rnn = CfC(n_features, hidden_size, batch_first=True, return_sequences=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        # ncps' current PyTorch release has an open broadcast defect for batched
        # `timespans` (its internal code squeezes B to shape [B]). Calling the
        # sequence layer one patient-window at a time preserves CfC's elapsed-time
        # semantics and makes this MVP work across those releases. Replace this
        # loop with one batched call after pinning a release that fixes the issue.
        outputs = []
        for sequence, elapsed in zip(x, delta_t):
            out_i, _ = self.rnn(sequence.unsqueeze(0), timespans=elapsed.unsqueeze(0))
            outputs.append(out_i)
        out = torch.cat(outputs, dim=0)
        return self.head(out[:, -1]).squeeze(-1)


def build_model(name: str, n_features: int, hidden_size: int, dropout: float):
    if name == "lstm":
        return LSTMClassifier(n_features, hidden_size, dropout)
    if name == "cfc":
        return CfCClassifier(n_features, hidden_size, dropout)
    raise ValueError("model must be 'lstm' or 'cfc'")
