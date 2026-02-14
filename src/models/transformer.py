import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformer_configuration import TransformerWMConfiguration as Config
import dummy_data_loader as DummyDataLoader
import math

"""
class WorldModel(nn.Module):
    def __init__(self, config=Config.TransformerWMConfiguration):
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
        """ """
        Roll out future latent states from initial state z0
        with sequence of future actions actions.
        
        :param self: -
        :param z0: (batch_size, 1, LATENT_DIM)
        :param actions: (batch_size, horizon length)
        :param returns: (batch_size, horizon length, LATENT_DIM)
        """ """
        BATCH_SIZE, SEQUENCE_LENGTH = actions.shape
        
        latent_hist = z0 # (batch_size, 1, 384)
        action_hist = []
        
        preds = []
        
        for t in range(SEQUENCE_LENGTH):
            a_t = actions[:, t:t+1] # (batch_size, 1)
            action_hist.append(a_t)
            
            a_seq = torch.cat(action_hist, dim=1)
            
            pred_latents, _, _ = self.forward(latent_hist, a_seq)
            
            z_next = pred_latents[:, -1:] # Take predicted step (B, 1, LATENT_DIM)
            
            latent_hist = torch.cat([latent_hist, z_next], dim=1)
            preds.append(z_next)
            
        return torch.cat(preds, dim=1) # (batch_size, horizon_length, LATENT_DIM)
"""
######################################

"""
Usage:

device = "cuda" if torch.cuda.is_available() else "cpu"
model = train_dummy_world_model_with_rollout(
    epochs=3,
    num_batches=5,
    batch_size=16,
    rollout_horizon=8,
    device=device
)
"""


"""
RoPE
Rotary Positional Embeddings for better extrapolation in autoregressive generation

"""

def apply_rope(x):
    """
    Rotates Q and K vectors according to timestep-dependent 
    angles so the transformer can understand relative position 
    in trajectories without adding positional embeddings.
    
    x: (batch_size, sequence_length, no. attention heads, head dimension (latent_dim / num_heads))
    """
    
    # Extract Dimensions
    batch_size, sequence_length, att_head_count, latent_dim = x.shape
    half = latent_dim // 2
    
    # Frequency Spectrum
    # Lower dims rotate slowly, higher dims rotate faster
    # Model learns multi-scale positional encoding
    freqs = torch.arange(half, device=x.device).float()
    freqs = 1.0 / (10000 ** (freqs / half))
    
    # Position indices
    pos = torch.arange(sequence_length, device=x.device).float()
    # Rotation angle for each time step and dimension
    angles = pos[:, None] * freqs[None, :]
    
    # Sin and cos tensors for rotation
    sin = torch.sin(angles)[None, :, None, :]
    cos = torch.cos(angles)[None, :, None, :]
    
    # Split feature vector into two halves and apply rotation
    # Shapes: (batch_size, sequence_length, att_head_count, half)
    x1 = x[..., :half]
    x2 = x[..., half:]
    
    # Apply rotation and recombine halves
    return torch.cat([x1 * cos - x2 * sin,
                     x1 * sin + x2 * cos], dim=-1)
    
    
"""
Causal Mask
"""

def causal_mask(sequence_length, device):
    """
    Returns a mask tensor
    .triu -> upper triangular part of a matrix turned to boolean mask
    True means masked out (no attention), False means allowed attention
    """
    return torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1).bool()


"""
Attention
Create Q, K, V
Applies RoPE
Compute causal self-attention
Combine heads
Project back to latent dim
"""

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        # Each head gets a portion of the latent vector dim
        # Eg, 384 with 6 heads = 64 dim per head
        self.heads = heads
        self.head_dim = dim // heads
        
        # Produce query, key, value
        self.qkv = nn.Linear(dim, dim * 3)
        # Projection after attention to merge the heads back to latent dim
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, sequence_length, latent_dim = x.shape
        
        # Create Q, K, V
        qkv = self.qkv(x)
        # Split
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Split into heads
        q = q.view(batch_size, sequence_length, self.heads, self.head_dim)
        k = k.view(batch_size, sequence_length, self.heads, self.head_dim)
        v = v.view(batch_size, sequence_length, self.heads, self.head_dim)
        
        # RoPE, positional information injected by rotating vectors
        q = apply_rope(q)
        k = apply_rope(k)
        
        # Attention scores, each token compares with every previous token
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask to prevent seeing future tokens
        mask = causal_mask(sequence_length, x.device)
        attn = attn.masked_fill(mask[None, :, None, :], -1e9)
        
        # Attention weights sum to 1 over past tokens
        attn = attn.softmax(dim=-1)
        
        # Weighted value aggregation
        # Each token's new representation is a weighted sum of value vectors from previous tokens
        out = attn @ v
        # Merge heads
        out = out.reshape(batch_size, sequence_length, latent_dim)
        # Final Projection
        return self.proj(out)
    
"""
Transformer
"""

class WMTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        
        # Normalize each token's latent vector before attention
        self.ln1 = nn.LayerNorm(dim)
        # Attention Module
        self.attn = Attention(dim, heads)
        
        # Normalize before MLP
        self.ln2 = nn.LayerNorm(dim)
        
        # Feedforward network with expansion (MLP ratio)
        # Eg: 384 -> 1536 -> 384 for MLP ratio of 4
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        
    def forward(self, x):
        # x: (batch_size, sequence_length, latent_dim)
        # Residual attention block (norm, attention, add)
        x = x + self.attn(self.ln1(x))
        # Residual MLP block (norm, MLP, add)
        x = x + self.mlp(self.ln2(x))
        return x
    
"""
World Model (Dino Style)
Predicts next latent, reward and value from previous latent sequence and action sequence
"""

class DinoWorldModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        
        # Action embedding to convert discrete actions into vectors that can be added to latents
        self.action_embed = nn.Embedding(config.ACTION_DIM,
                                         config.LATENT_DIM)
        
        # Transformer blocks for processing the sequence of latents and actions
        self.blocks = nn.ModuleList([
            WMTransformerBlock(config.LATENT_DIM,
                   config.NUM_HEADS,
                   config.MLP_RATIO)
            for _ in range(config.NUM_LAYERS)
        ])
        
        # Layer norm before output heads
        self.ln_f = nn.LayerNorm(config.LATENT_DIM)
        
        # Predict change in latent (delta) rather than absolute next latent for better stability
        self.delta_head = nn.Linear(config.LATENT_DIM,
                                    config.LATENT_DIM)
        
        # Predict reward and value for each time step in the sequence
        self.reward_head = nn.Linear(config.LATENT_DIM, 1)
        self.value_head = nn.Linear(config.LATENT_DIM, 1)
        
    # Delta-latent prediction
    
    def forward(self, latents, actions):
        """
        Dynamics Prediction
        
        :param self: -
        :param latents: (batch_size, sequence_length, latent_dim)
        :param actions: (batch_size, sequence_length) - discrete action indices
        :returns: pred_next_latents (batch_size, sequence_length, latent_dim),
        """
        
        # Combine latent + action embeddings as input to transformer
        x = latents + self.action_embed(actions)
        
        # Model attends to past, learns temporal dependencies and builds world representation
        for blk in self.blocks:
            x = blk(x)
            
        # Final norm before heads
        x = self.ln_f(x)
        
        # Delta prediction: model learns to predict how the latent state changes rather than absolute next state, which is often easier to learn and more stable
        delta = self.delta_head(x)
        pred_next_latents = latents + delta
        
        # Reward and value predictions for each time step, useful for training and planning
        pred_rewards = self.reward_head(x)
        pred_values = self.value_head(x)
        
        return pred_next_latents, pred_rewards, pred_values
        
    # Rollout (history-aware imagination)
    
    @torch.no_grad()
    def rollout(self, z0, actions):
        """
        Imagination engine
        
        :param self: -
        :param z0: Initial latent state (batch_size, 1, latent_dim)
        :param actions: Sequence of future actions (batch_size, sequence_length)
        :returns: pred_latents (batch_size, sequence_length, latent_dim),
        """
        
        batch_size, sequence_length = actions.shape
        
        latent_hist = z0
        action_hist = []
        
        preds_latents = []
        preds_rewards = []
        preds_values = []
        
        #
        for t in range(sequence_length):
            # Add aciton to history
            a_t = actions[:, t:t+1] # (batch_size, 1)
            action_hist.append(a_t)
            
            # Concatenate actions so far for autoregressive input
            a_seq = torch.cat(action_hist, dim=1)
            
            # Predict entire sequence of future latents, rewards and values given history so far
            pred_latents_seq, pred_rewards_seq, pred_values_seq = self.forward(latent_hist, a_seq)
            
            # Take last predicted time step
            z_next = pred_latents_seq[:, -1:] # (batch_size, 1, 384)
            r_next = pred_rewards_seq[:, -1:] # (batch_size, 1, 1)
            v_next = pred_values_seq[:, -1:] # (batch_size, 1, 1)
            
            # Append predictions
            preds_latents.append(z_next)
            preds_rewards.append(r_next)
            preds_values.append(v_next)
            
            # Append new latent to history for next step
            latent_hist = torch.cat([latent_hist, z_next], dim=1)
        
        # Concatenate over time
        pred_latents = torch.cat(preds_latents, dim=1) # (batch_size, sequence_length, latent_dim) 
        pred_rewards = torch.cat(preds_rewards, dim=1) # (batch_size, sequence_length, 1)
        pred_values = torch.cat(preds_values, dim=1) # (batch_size, sequence_length, 1)
        return pred_latents, pred_rewards, pred_values
    
    
    """
    When rolling out multiple candidate action sequences, 
    we can batch them together for efficiency. This method 
    takes an initial latent state and a batch of candidate 
    action sequences, and returns the predicted latents, rewards 
    and values for all candidates in one forward pass.
    """
    @torch.no_grad()
    def rollout_candidates(self, z0, actions_candidates):
        """
        Rollout multiple trajectories in parallel for CEM planning
        
        :param self: -
        :param z0: Initial latent state (batch_size, 1, latent_dim)
        :param actions_candidates: Batch of candidate action sequences (batch_size, candidate_sequence_count, sequence_length)
        :returns: pred_latents (batch_size, candidate_sequence_count, sequence_length, latent
        """
        
        batch_size, candidate_sequence_count, sequence_length = actions_candidates.shape
        device = z0.device
        
        z0_expanded = z0.unsqueeze(1).expand(batch_size, candidate_sequence_count, 1, z0.size(-1))
        z0_expanded = z0_expanded.reshape(batch_size * candidate_sequence_count, 1, -1)
        
        actions_flat = actions_candidates.reshape(batch_size * candidate_sequence_count, sequence_length)
        
        pred_latents, pred_rewards, pred_values = self.rollout(z0_expanded, actions_flat)
        
        pred_latents = pred_latents.reshape(batch_size, candidate_sequence_count, sequence_length, -1)
        pred_rewards = pred_rewards.reshape(batch_size, candidate_sequence_count, sequence_length, 1)
        pred_values = pred_values.reshape(batch_size, candidate_sequence_count, sequence_length, 1)
        
        return pred_latents, pred_rewards, pred_values


"""
MSE Loss Function
"""

def latent_mse_loss(pred_latents, target_latents):
    """
    Calculate mean squared error between predicted and target latents for training the world model.
    
    :param pred_latents: Predicted next latents from the model (batch_size, sequence_length, latent_dim)
    :param target_latents: Ground truth next latents (batch_size, sequence_length, latent_dim)
    """
    
    return F.mse_loss(pred_latents, target_latents)


"""
Score action sequences
"""
def score_action_sequences(pred_rewards, gamma=0.99):
    """
    Compute discounted returns for each candidate action sequence based on predicted rewards.
    
    :param pred_rewards: Predicted rewards for each time step in the sequence (batch_size, candidate_sequence_count, sequence_length, 1)
    :param gamma: Discount factor for future rewards
    """
    
    batch_size, candidate_sequence_count, sequence_length, _ = pred_rewards.shape
    device = pred_rewards.device
    
    # Discount tensor
    discounts = torch.tensor(
        [gamma**t for t in range(sequence_length)],
        device=device
    ).view(1, 1, sequence_length, 1)
    
    # Apply discounts to predicted rewards and sum over time to get return for each candidate sequence
    discounted = pred_rewards * discounts
    returns = discounted.sum(dim=2)
    
    # One score per candidate sequence, shape (batch_size, candidate_sequence_count, 1)
    # Squeeze last dimension to get (batch_size, candidate_sequence_count)
    return returns.squeeze(-1) # (batch_size, candidate_sequence_count)

"""
CEM Planner
"""
class CEMPlanner:
    def __init__(
        self,
        model,
        action_dim=4,
        horizon=8,
        num_candidates=64,
        num_elites=8,
        num_iters=4,
        gamma=0.99,
        device=torch.device("cpu")
    ):
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.num_iters = num_iters
        self.gamma = gamma
        self.device = device
    """
    model: world model
    action_dim: 4
    horizon: 16
    num_candidates: tbd
    num_elites: top sequences to use to update the distribution
    num_iters: number of CEM iterations
    gamma: discount factor for returns
    """
    @torch.no_grad()
    def plan(self, z0):
        """
        z0: (B, 1, LATENT_DIM)

        Returns:
            best_action: (B,)
        """
        
        # Extraxt batch size
        batch_size = z0.size(0)

        # Initialize uniform categorical probabilities
        probs = torch.ones(
            batch_size, self.horizon, self.action_dim,
            device=self.device
        ) / self.action_dim

        # CEM iterations
        for _ in range(self.num_iters):

            # Sample action sequences
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample((self.num_candidates,))  
            # (number of candidates, batch_size, sequence_length)

            actions = actions.permute(1, 0, 2)  
            # (number of candidates, batch_size, sequence_length)

            # Rollout all candidates
            pred_latents, pred_rewards, pred_values = \
                self.model.rollout_candidates(z0, actions)

            # Score sequences
            returns = score_action_sequences(pred_rewards, gamma=self.gamma)
            # (batch_size, number of candidates)

            # Select elites
            elite_idx = returns.topk(self.num_elites, dim=1).indices
            # (batch_size, num_elites)

            elite_actions = torch.gather(
                actions,
                1,
                elite_idx.unsqueeze(-1).expand(-1, -1, self.horizon)
            )
            # (batch_size, num_elites, sequence_length)

            # Update distribution
            new_probs = torch.zeros_like(probs)

            for a in range(self.action_dim):
                new_probs[:, :, a] = (elite_actions == a).float().mean(dim=1)

            probs = new_probs.clamp(min=1e-4)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Choose best sequence
        dist = torch.distributions.Categorical(probs=probs)
        best_sequence = dist.sample()  # (B, H)

        # Return first action (MPC style)
        return best_sequence[:, 0]

"""
MPC Controller
Stores a planner instance and uses it to select actions based on the current latent state.
"""
class MPCController:
    def __init__(self, planner):
        self.planner = planner

    @torch.no_grad()
    def act(self, z0):
        """
        z0: (batch_size, 1, LATENT_DIM)
        Returns:
            action: (batch_size,)
        """
        return self.planner.plan(z0)



"""
Training Step
"""

def train_step(model, optimizer, latents, actions, device=torch.device("cpu")):
    """
    Single training step for world model using MSE loss on predicted next latents.
    
    :param model: Dino World Model
    :param optimizer: Adam?
    :param latents: (batch_size, sequence_length, latent_dim) - input latent sequences
    :param actions: (batch_size, sequence_length) - input action sequences
    :param device: device
    """
    
    model.train()
    optimizer.zero_grad()
    
    latents = latents.to(device)
    actions = actions.to(device)
    
    # Shifted sequences for next-step prediction
    z_in = latents[:, :-1]
    a_in = actions[:, :-1]
    z_target = latents[:, 1:]
    
    # Forward pass
    pred_next_latents, pred_rewards, pred_values = model(z_in, a_in)
    
    # Compute latent MSE
    loss_latent = latent_mse_loss(pred_next_latents, z_target)
    # loss_reward = F.mse_loss(pred_rewards, reward_targets)
    # loss_value = F.mse_loss(pred_values, value_targets)
    
    # Optional reward value losses
    loss = loss_latent # + loss_reward + loss_value
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

"""
Training loop with dummy data and rollout testing
"""

def train_dummy_world_model(epochs=5, 
                            num_batches=10, 
                            batch_size=32, 
                            rollout_horizon=16,
                            device=Config.DEVICE):
    print("Starting Dummy Training")
    
    # Initilizse
    config = Config()
    model = DinoWorldModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Epoch loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Batch loop with dummy data
        for latents, actions in DummyDataLoader.dummy_loader(num_batches, batch_size, device):
            loss_val = train_step(model, optimizer, latents, actions, device)
            epoch_loss += loss_val
            
        # Avg loss
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss/num_batches:.4f}")

        # Imagination / Rollout Test
        z0 = torch.randn(batch_size, 1, Config.LATENT_DIM, device=device)
        
        # future_actions = torch.randint(0, Config.ACTION_DIM, (batch_size, rollout_horizon), device=device)
        # 5 sequences per batch for testing
        future_actions_candidates = torch.randint(0, Config.ACTION_DIM, (batch_size, 5, rollout_horizon))
        
        # Single
        #pred_latents, pred_rewards, pred_values = model.rollout(z0, future_actions)
        
        # Multi candidates
        pred_latents, pred_rewards, pred_values = model.rollout_candidates(z0, future_actions_candidates)
        
        # Score candidates
        returns = score_action_sequences(pred_rewards)
        # Best candidate per batch
        best_id = returns.argmax(dim=1)
        
        print("Returns shape:", returns.shape)     # (B, N)
        print("Best candidate per batch:", best_id)
        
        print(f"Rollout shapes | latents: {pred_latents.shape}, rewards: {pred_rewards.shape}, values: {pred_values.shape}")
        
    print("=== Dummy Training Completed ===")
    return model
     
            
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
evaluate_dummy_world_model(model, batch_size=16, rollout_horizon=10, device=device)
"""            


def evaluate_dummy_world_model(model, batch_size=32, rollout_horizon=16, device=torch.device("cpu")):
    """
    Evaluate trained world model with dummy data
    
    :param model: Trained world model to evaluate
    :param batch_size: Number of parallel rollouts to test
    :param rollout_horizon: Number of time steps to rollout into the future
    :param device: device
    """
    
    model.eval()
    
    # Generate dummy initial latent and action states for rollout
    z0 = torch.randn(batch_size, 1, Config.LATENT_DIM, device=device)
    future_actions = torch.randint(0, Config.ACTION_DIM, (batch_size, rollout_horizon), device=device)
    
    # Rollout future latents, rewards and values from the model
    pred_latents, pred_rewards, pred_values = model.rollout(z0, future_actions)
    
    assert pred_latents.shape == (batch_size, rollout_horizon, Config.LATENT_DIM), "Latent shape mismatch"
    assert pred_rewards.shape == (batch_size, rollout_horizon, 1), "Reward shape mismatch"
    assert pred_values.shape == (batch_size, rollout_horizon, 1), "Value shape mismatch"

    
    print("✅ Rollout shapes correct:")
    print(f"Latents: {pred_latents.shape}, Rewards: {pred_rewards.shape}, Values: {pred_values.shape}")

    # --- Quick numerical sanity check ---
    print("\n--- Latent stats ---")
    print(f"min: {pred_latents.min():.3f}, max: {pred_latents.max():.3f}, mean: {pred_latents.mean():.3f}")
    
    print("\n--- Reward stats ---")
    print(f"min: {pred_rewards.min():.3f}, max: {pred_rewards.max():.3f}, mean: {pred_rewards.mean():.3f}")
    
    print("\n--- Value stats ---")
    print(f"min: {pred_values.min():.3f}, max: {pred_values.max():.3f}, mean: {pred_values.mean():.3f}")

    print("\n✅ Sanity check passed: no crashes, shapes and stats look reasonable.\n")
    
"""
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, latents, actions = overfit_single_batch(
        batch_size=32,
        sequence_length=16,
        epochs=200,
        device=device
    )
"""    

def overfit_single_batch(
    model=None,
    batch_size=32,
    sequence_length=16,
    latent_dim=Config.LATENT_DIM,
    action_dim=Config.ACTION_DIM,
    lr=1e-3,
    epochs=200,
    device=Config.DEVICE
):
    """
    Overfits the world model on a single random batch.
    Loss should approach 0 if everything works.
    """
    
    device = torch.device(device)
    
    # Generate single dummy batch
    latents = torch.randn(batch_size, sequence_length, latent_dim, device=device)
    actions = torch.randint(0, action_dim, (batch_size, sequence_length), device=device)
    
    # Initialize model if not provided
    if model is None:
        config = Config()
        model = DinoWorldModel(config).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("Starting overfit on single batch...")
    
    for epoch in range(epochs):
        loss_val = train_step(model, optimizer, latents, actions, device)
        
        # Logging
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_val:.6f}")
        
        # Optional early stop
        if loss_val < 1e-6:
            print(f"Early stopping at epoch {epoch+1}, loss < 1e-6")
            break
    
    # Rollout check
    model.eval()
    with torch.no_grad():
        z_in = latents[:, :-1]
        a_in = actions[:, :-1]
        pred_latents, pred_rewards, pred_values = model(z_in, a_in)
        
        mse = F.mse_loss(pred_latents, latents[:, 1:]).item()
        print(f"\nRollout MSE on single batch after overfit: {mse:.8f}")
        
        # Optional: print min/max/mean differences
        diff = pred_latents - latents[:, 1:]
        print(f"Latent diff | min: {diff.min():.6f}, max: {diff.max():.6f}, mean: {diff.mean():.6f}")
    
    
    print("=== Overfit + Rollout completed ===")
    return model, latents, actions, pred_latents, pred_rewards, pred_values

"""
Overfit multiple batches
"""    
def demo_overfit_mpc(model, batch_size=4, horizon=5, num_candidates=16, num_elites=4, device=Config.DEVICE):
    """
    Demonstrates MPC planning using CEM on an overfitted world model.
    """
    device = torch.device(device)
    
    model.eval()
    
    # Initial latent (from overfit batch or random)
    z0 = torch.randn(batch_size, 1, Config.LATENT_DIM, device=device)
    
    # Initialize CEM planner
    planner = CEMPlanner(
        model=model,
        action_dim=Config.ACTION_DIM,
        horizon=horizon,
        num_candidates=num_candidates,
        num_elites=num_elites,
        num_iters=5,
        gamma=0.99,
        device=device
    )
    
    # Wrap planner in MPC controller
    mpc = MPCController(planner)
    
    # Select action
    action = mpc.act(z0)
    
    print("=== MPC / CEM Demo ===")
    print(f"Initial latent shape: {z0.shape}")
    print(f"Selected action per batch: {action}")  # (B,)
    print(f"Action type: {action.dtype}, min: {action.min().item()}, max: {action.max().item()}")
    print("======================\n")
    
    return action


    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_dummy_world_model(
        epochs=3,
        num_batches=5,
        batch_size=32,
        rollout_horizon=16,
        device=device
    )
    
    evaluate_dummy_world_model(model,
                               batch_size=32,
                               rollout_horizon=16,
                               device=device)