import torch
import torch.nn as nn
import torch.optim as optim

# --- MOCK IMPORTS (Eventually these will be real imports) ---
# from src.data.buffer import ReplayBuffer
# from src.models.transformer import WorldModel

class MockBuffer:
    """Simulates Member C's Data Buffer"""
    def sample(self, batch_size):
        # Returns random tensors matching the Interface Contract
        # Shape: (Batch, Seq_Len, Latent_Dim)
        return torch.randn(batch_size, 16, 384), torch.randint(0, 4, (batch_size, 16))

class MockModel(nn.Module):
    """Simulates Member B's Transformer"""
    def __init__(self):
        super().__init__()
        # A simple linear layer that has "learnable weights"
        # It takes size 384 and outputs size 384
        self.linear = nn.Linear(384, 384) 

    def forward(self, latents, actions):
        batch_size = latents.shape[0]
        
        # FIX: Pass the input through the layer!
        # Now 'pred_next_latents' is connected to 'self.linear.weight'
        pred_next_latents = self.linear(latents)
        
        # For rewards, we can still return random noise or a simple projection
        # Since we aren't training on reward loss yet in the skeleton
        pred_rewards = torch.randn(batch_size, 16, 1).to(latents.device)
        
        return pred_next_latents, pred_rewards
# --- CONFIGURATION ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10

def train():
    print("Starting Training Loop (SKELETON MODE)...")
    
    # 1. Initialize Components (Using Mocks for now)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    buffer = MockBuffer()
    model = MockModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # 2. Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        
        # Simulate one batch of training
        optimizer.zero_grad()
        
        # A. Load Data (The Handshake with Member C)
        latents, actions = buffer.sample(BATCH_SIZE)
        latents, actions = latents.to(device), actions.to(device)
        
        # B. Forward Pass (The Handshake with Member B)
        # We predict the *next* latent state based on current one
        pred_next_latents, pred_rewards = model(latents, actions)
        
        # C. Calculate Loss (Simple MSE for now)
        # In reality, we compare pred_next_latents[:, :-1] vs latents[:, 1:]
        loss = criterion(pred_next_latents, latents) 
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        print(f"   Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f}")

    print("Integration Test Passed: The loop runs without crashing.")

if __name__ == "__main__":
    train()