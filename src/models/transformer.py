import torch
import torch.nn as nn

class Configuration:
    LATENT_DIM = 384
    SEQUENCE_LENGTH = 16
    NUM_LAYERS = 4
    NUM_HEADS = 4
    ACTION_DIM = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")

class WorldModel(nn.Module):
    def __init__(self, config=Configuration):
        super().__init__()
        self.action_embed = nn.Embedding(config.ACTION_DIM, config.ACTION_EMBED_DIM)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.LATENT_DIM,
                nhead=config.NUM_HEADS,
                batch_first=True
            ),
            num_layers=config.NUM_LAYERS
        )
        
        self.next_latent_head = nn.Linear(config.LATENT_DIM, config.LATENT_DIM)
        self.reward_head = nn.Linear(config.LATENT_DIM, 1)
        self.value_head = nn.Linear(config.LATENT_DIM, 1)
        
    def forward(self, latents, actions):
        a_embed = self.action_embed(actions)
        x = latents + a_embed
        
        h = self.transformer(x)
        
        pred_next_latents = self.next_latent_head(h) # (batch_size, SEQUENCE_LENGTH, 384)
        pred_rewards = self.reward_head(h) # (batch_size, SEQUENCE_LENGTH, 1)
        pred_values = self.value_head(h)
        
        return pred_next_latents, pred_rewards, pred_values
    
    @torch.no_grad()
    def rollout(self, z0, actions):
        """
        Roll out future latent states from initial state z0
        with sequence of future actions actions.
        
        :param self: -
        :param z0: (batch_size, 1, LATENT_DIM)
        :param actions: (batch_size, horizon length)
        :param returns: (batch_size, horizon length, LATENT_DIM)
        """
        BATCH_SIZE, SEQUENCE_LENGTH = actions.shape
        
        preds = []
        z_t = z0
        
        for t in range(SEQUENCE_LENGTH):
            a_t = actions[:, t:t+1] # (batch_size, 1)
            pred_latents, _, _ = self.forward(z_t, a_t)
            z_t = pred_latents[:, -1:] # Take predicted step (B, 1, LATENT_DIM)
            preds.append(z_t)
            
        return torch.cat(preds, dim=1) # (batch_size, horizon_length, LATENT_DIM)
            