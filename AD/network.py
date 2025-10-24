import torch
import torch.nn as nn
import numpy

class ADTransformer(nn.Module):

    def __init__(self, state_dim, action_dim, n_embd=128, n_layer=4, n_head=4, dropout=0.1, max_seq_len=50000):
        super().__init__()
        self.n_embd = n_embd
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Embeddings for each modality
        self.state_emb = nn.Linear(state_dim, n_embd)
        self.action_emb = nn.Linear(action_dim, n_embd)
        self.reward_emb = nn.Linear(1, n_embd)

        # Type embedding: 0=state, 1=action, 2=reward
        self.type_emb = nn.Embedding(3, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)

        # Transformer encoder-decoder (causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, action_dim)



    def forward(self, states, actions, rewards):

        B, T, E = states.shape
        device = states.device

        state_tokens = self.state_emb(states)
        action_tokens = self.action_emb(actions)
        reward_tokens = self.reward_emb(rewards.unsqueeze(-1))

        stacked_inputs = torch.stack([state_tokens[:, :-1], action_tokens, reward_tokens], dim=2)
        interleaved = stacked_inputs.reshape(B, (T - 1) * 3, self.n_embd)
        tokens = torch.cat([interleaved, state_tokens[:, -1].unsqueeze(1)], dim=1)

        type_seq = []
        for t in range(T - 1):
            type_seq.extend([0, 1, 2])
        type_seq.append(0)
        type_seq = torch.tensor(type_seq, device=device).unsqueeze(0).expand(B, -1)
        pos = torch.arange(tokens.size(1), device=device).unsqueeze(0).expand(B, -1)

        x = tokens + self.type_emb(type_seq) + self.pos_emb(pos)

        L = x.size(1)
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()  # upper triangular
        x = self.transformer(x, mask=causal_mask)

        x = self.norm(x)

        state_idx = torch.arange(0, L, 3, device=device)
        state_reps = x[:, state_idx, :]
        output = self.head(state_reps)

        return output
