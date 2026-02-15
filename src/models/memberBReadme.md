## dummy_data_loader.py (B.0)

### dummy_loader

Creates a number of batches of certain size, purely dummy data. Can do images (default=false), latents (default=true), actions (default=true) and rewards (default=true).

Has some CONSTANTS at the top of the file

Use dummy_loader(num_batches, batch_size, device).

## transformer_configuration.py (B.1)

Class TransformerWMConfiguration that has a bunch of parameters. Essentially a bunch of constants.

## Main File: transformer.py

apply_rope(x) - Rotary Positional Embeddings. See code comments.

causal_mask(horizon length, device) - Uses torch.triu

### Attention

Takes in batch size, horizon length and latent space dimension. Creates query, keys and values. Applies RoPE to Q and K, computes attention scores for those two. Applies causal mask. Read code comments.

### World Model Transformer

Applies a norm layer, attention, then norm layer, MLP. Code comments.

### Dino World Model

init()

action_embed converts discrete actions into vectors that can be added to latents.

blocks - transformer blocks for processing the sequence of latents and actions

ln_f - layer norm before output heads

delta_head - predict change in latent rather than absolute next latent

reward_head and value_head - predict reward and value for each time step in sequence

forward() - predicts next latents, rewards and values

rollout(z0, actions) - Rollouts 1 sequence given an initial latent state and a sequence of actions

rollout_candidates(z0, actions_candidates) - same thing but does multiple sequences at once.

### Loss function

Simple latent_mse_loss, latent_mse_loss(pred_latents, target_latents).

### Score Action Sequences

Computes discounted returns for each candidate action sequence

### CEM Planner

Params: action dimension, horizon length, number of candidates, number of elites (best trajectories), number of iterations (to repeat this process), gamma

plan(z0) - Returns the first action in the best sequence

### MPC Controller

Stores a planner instance and uses it to select actions based on current latent state.

### Training Step

train_step(model, optimizer, latents, actions, device) - single training step

### Training Dummy World Model

train_dummy_world_model(epochs, num_batches, batch_size, horizon_length) - Trains the dummy world model

### Evaluate Dummy World Model

evaluate_dummy_world_model(model, batch_size, horizon_length, device) - Evaluates the model

### Single Batch

overfit_single_batch(model, batch_size, horizon_length, latent_dim, action_dim, learning_rate, epochs, device) - Overfits world model on a single random batch. Loss should approach 0 if functions correctly.

### Overfit Multiple Batches

demo_overfit_mpc(model, batch_size, horizon_length, number of candidates, number of elites, device) - Overfit multiple batches





