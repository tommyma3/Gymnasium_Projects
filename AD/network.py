import torch
import torch.nn as nn

class ADTransformerInterleaved(nn.Module):
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
            d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, action_dim)

    def forward(self, states, actions, rewards):
        """
        states:  (B, T, state_dim)
        actions: (B, T, action_dim)
        rewards: (B, T)
        returns: (B, action_dim) predicted next action after the sequence
        """
        B, T, _ = states.shape
        device = states.device

        tokens, token_types = [], []

        for t in range(T):
            tokens.append(self.state_emb(states[:, t, :]))
            token_types.append(torch.zeros(B, dtype=torch.long, device=device))  # state

            if t < T - 1:
                tokens.append(self.action_emb(actions[:, t, :]))
                token_types.append(torch.ones(B, dtype=torch.long, device=device))  # action

                tokens.append(self.reward_emb(rewards[:, t].unsqueeze(-1)))
                token_types.append(2 * torch.ones(B, dtype=torch.long, device=device))  # reward

        x = torch.stack(tokens, dim=1)
        token_types = torch.stack(token_types, dim=1)

        type_embedding = self.type_emb(token_types)
        positions = torch.arange(x.size(1), device=device).unsqueeze(0).expand(B, -1)
        pos_embedding = self.pos_emb(positions)

        x = x + type_embedding + pos_embedding


        # Type and position embeddings
        type_embedding = self.type_emb(token_types.long())
        positions = torch.arange(x.size(1), device=device).unsqueeze(0).expand(B, -1)
        pos_embedding = self.pos_emb(positions)

        x = x + type_embedding + pos_embedding

        # Causal mask: lower-triangular (prevent looking ahead)
        L = x.size(1)
        causal_mask = torch.tril(torch.ones(L, L, device=device))
        x = self.transformer(x, mask=(~causal_mask.bool()))

        x = self.norm(x)

        # Predict next action based on the last state token
        # Find last state token index: every 3 tokens (s, a, r)
        last_state_idx = 3 * T - 3  # last s_T-1
        output = self.head(x[:, last_state_idx, :])

        return output
