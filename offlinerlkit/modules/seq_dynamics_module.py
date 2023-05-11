import torch
import torch.nn as nn
from typing import Optional

from offlinerlkit.nets import DecoderBlock


class GPTDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        seq_len: int = 10,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        device: str='cpu',
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(2*seq_len, embedding_dim)
        self.state_emb = nn.Linear(obs_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    seq_len=2 * seq_len,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.state_head = nn.Linear(embedding_dim, 2*obs_dim)
        self.reward_head = nn.Linear(embedding_dim, 1)

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.apply(self._init_weights)
        self.to(device)
        self.device = device

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_steps = torch.arange(0, seq_len, device=states.device).repeat(batch_size,1)
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb

        # [batch_size, seq_len * 2, emb_dim], ( s_0, a_0, s_1, a_1, ...)
        sequence = (
            torch.stack([state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 2 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        out_state = self.state_head(out[:, 1::2])
        reward = self.reward_head(out[:, 1::2])
        mu, logvar = torch.chunk(out_state, 2, -1)
        logvar = torch.clamp(logvar, -10, 1)
        return mu, logvar, reward