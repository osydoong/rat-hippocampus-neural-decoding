import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden = max(input_dim, latent_dim)      
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # in: [N, K, M]  →  out: [N, K, d]
        return self.net(x)


class RecurrentBlock(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int,
                 rnn_type: str = 'GRU', num_layers: int = 1):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.upper() == 'GRU' else nn.RNN
        self.rnn = rnn_cls(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_dim = hidden_dim

    def forward(self, packed_input):
        packed_out, _ = self.rnn(packed_input)
        # pad_packed_sequence: PackedSequence → [N, K_max, hidden_dim]
        output, lengths = pad_packed_sequence(packed_out, batch_first=True)
        return output, lengths


class MLPRegressor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h):
        # h: [N, K, hidden_dim]  →  out: [N, K]
        return self.net(h).squeeze(-1)


class RatModel(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 64,
                 hidden_dim: int = 64, rnn_type: str = 'GRU',
                 num_layers: int = 1):
        super().__init__()
        self.encoder   = MLPEncoder(input_dim, latent_dim)
        self.recurrent = RecurrentBlock(latent_dim, hidden_dim, rnn_type, num_layers)
        self.regressor = MLPRegressor(hidden_dim)

    def forward(self, x, lengths):
        # 1. Encode هر timestep
        z = self.encoder(x)                             # [N, K_max, d]

        # 2. Pack → RNN → Unpack
        packed = pack_padded_sequence(
            z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _ = self.recurrent(packed)                   # [N, K_max, hidden]

        # 3. Regress position
        out = self.regressor(h)                         # [N, K_max]
        return out


class SharedRNNModel(nn.Module):

    def __init__(self, rat_dims: dict, latent_dim: int = 64,
                 hidden_dim: int = 64, rnn_type: str = 'GRU',
                 num_layers: int = 1):
        super().__init__()

        # Encoder جداگانه برای هر موش
        self.encoders = nn.ModuleDict({
            rat: MLPEncoder(dim, latent_dim)
            for rat, dim in rat_dims.items()
        })

        # RNN و Regressor مشترک
        self.recurrent = RecurrentBlock(latent_dim, hidden_dim, rnn_type, num_layers)
        self.regressor  = MLPRegressor(hidden_dim)

    def forward(self, x, lengths, rat_name: str):
        
        z      = self.encoders[rat_name](x)             # [N, K_max, d]
        packed = pack_padded_sequence(
            z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h, _   = self.recurrent(packed)                 # [N, K_max, hidden]
        out    = self.regressor(h)                      # [N, K_max]
        return out

    def freeze_shared(self):
        
        for param in self.recurrent.parameters():
            param.requires_grad = False
        for param in self.regressor.parameters():
            param.requires_grad = False

    def unfreeze_shared(self):
        for param in self.recurrent.parameters():
            param.requires_grad = True
        for param in self.regressor.parameters():
            param.requires_grad = True

    def add_rat_encoder(self, rat_name: str, input_dim: int):
        
        self.encoders[rat_name] = MLPEncoder(input_dim, 
                                             self.recurrent.rnn.input_size)
