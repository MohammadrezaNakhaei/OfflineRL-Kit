from offlinerlkit.nets import DecoderBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import MutableMapping, Optional, Tuple
import numpy as np

class ContextEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        embedding_dim: int = 64,
        output_dim: int = 16,
        num_layers: int = 1,
        num_heads: int = 4,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding( seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    seq_len = 2*seq_len,
                    embedding_dim = embedding_dim,
                    hidden_dim = embedding_dim*4,
                    num_heads = num_heads,
                    attention_dropout = attention_dropout,
                    residual_dropout = residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_head = nn.Linear(embedding_dim, output_dim)
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.apply(self._init_weights)

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
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb

        sequence = (
            torch.stack([state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            padding_mask = (
                torch.stack([padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 2 * seq_len)
            )
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)
        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)
        out = self.out_norm(out)
        out = self.out_head(out[:, -2]) # condition on last state, not last action 
        return out
    
class ContextPredictor(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, latent_dim:int, hidden_dim:Tuple[int]=(256, 256)):
        super().__init__()
        n_layers = [state_dim+action_dim+latent_dim, *hidden_dim, state_dim]
        layers = []
        for l1, l2 in zip(n_layers[:-1], n_layers[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.ReLU())
        layers.pop()
        self.fc = nn.Sequential(*layers)

    def forward(self, states:torch.Tensor, actions:torch.Tensor, latents:torch.Tensor):
        total_input = torch.cat([states, actions, latents], dim=-1)
        return self.fc(total_input)


class EncoderModule():
    def __init__(self, encoder: ContextEncoder, predictor: ContextPredictor, lr:float=1e-4, alpha_sim:float=0.2, beta_dif:float=0.1):
        self.encoder = encoder
        self.predictor = predictor
        self.optimizer = torch.optim.Adam(list(encoder.parameters())+list(predictor.parameters()), lr)
        self.alpha_sim = alpha_sim
        self.beta_dif = beta_dif
    
    def __call__(self, seq_state:torch.Tensor, seq_action:torch.Tensor, time_step:torch.Tensor):
        return self.encode(seq_state, seq_action, time_step)

    @torch.no_grad()
    def encode(self, seq_state:torch.Tensor, seq_action:torch.Tensor, time_step:torch.Tensor):
        assert seq_state.ndim==3 # batch, seq, state
        return self.encoder(seq_state, seq_action, time_step)
    
    def encode_multiple(self, seq_states:torch.Tensor, seq_actions:torch.Tensor, seq_masks:torch.Tensor):
        assert seq_states.ndim==4
        B, N, T, _ = seq_states.shape
        device = seq_states.device
        timesteps = torch.arange(0, T, device=device).repeat(N*B, 1)
        latents = self.encoder(
            seq_states.view(B*N, T, -1), 
            seq_actions.view(B*N, T, -1),
            timesteps,
            seq_masks.view(B*N, T).bool(),
            )
        latents = latents.view(B, N, -1)
        return latents           
    
    def learn_batch(self, batch:MutableMapping[str, torch.Tensor], k=None):
        # currently no contrastive loss, only consider the mean of latent space
        seq_states = batch['seq_states']
        seq_actions = batch['seq_actions']
        seq_masks = batch['seq_masks']
        state = batch['state']
        action = batch['action']
        next_state = batch['next_state']
        latents = self.encode_multiple(seq_states, seq_actions, seq_masks)
        B, N, T, _ = seq_states.shape
        if k is None:
            k = T//2
        assert k<T
        # states = seq_states[:, :, 0:-1].reshape(B*N*(T-1),-1)
        # actions = seq_actions[:, :, 0:-1].reshape(B*N*(T-1),-1)
        # next_states = seq_states[:, :, 1:].reshape(B*N*(T-1),-1)
        idx = np.random.randint(N)
        predicted_state = self.predictor(state, action, latents[:,idx])
        # predicted_states = self.predictor(states, actions, latents.mean(1).repeat(N*(T-1),1))

        self.optimizer.zero_grad()
        # loss = cos(predicted_state, next_state-state).sum()
        loss_sim = similarity_loss(latents) # similar to N points
        loss_pred = F.mse_loss(predicted_state, next_state-state)
        
        # kstep prediction loss
        state = seq_states[:, :, 0,]
        for j in range(k):
            pred_diff = self.predictor(state, seq_actions[:, :, j], latents)
            loss_pred += F.mse_loss(pred_diff, seq_states[:, :, j+1,]-seq_states[:, :, j,])
            state = pred_diff+state


        # loss_pred_seq = F.mse_loss(predicted_states, next_states-states)
        loss = loss_pred + self.alpha_sim*loss_sim 
        loss.backward()
        self.optimizer.step()
        return {
            'loss/encoder_prediction': loss_pred.item(), 
            'loss/encoder_similarity':loss_sim.item(),
            # 'loss/encoder_prediction_seq': loss_pred_seq.item(),
            }
    
    def train(self, ):
        self.encoder.train()
        self.predictor.train()

def cosine(pred, target, reduce=False):
    x = F.normalize(pred, dim=-1, p=2)
    y = F.normalize(target, dim=-1, p=2)

    return 2 - 2*(x * y).sum(dim=-1, keepdim=(not reduce))

def similarity_loss(latents:torch.Tensor):
    assert latents.ndim == 3
    B, N, _ = latents.shape
    assert N>1
    device = latents.device
    latents = F.normalize(latents, dim=-1, p=2)
    masks = torch.eye(N, device = device).unsqueeze(0).repeat(B, 1, 1)
    sum_similarity = latents@latents.permute(0, 2, 1) # B, N, N
    sum_similarity = (1-masks)*sum_similarity
    loss = -1/(N-1)*sum_similarity.sum(dim=(1,2)).mean()
    return loss
