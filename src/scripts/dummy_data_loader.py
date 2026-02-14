"""
# B.0

Generate and Load Dummy Data for Training

Example Usage
for latents, actions in dummy_loader(50, 32, return_images=False):
    print(latents.shape)  # (32, 16, 384)
    print(actions.shape)  # (32, 16, 4)
    
""" 

import torch

SEQUENCE_LENGTH = 16 # Horizon
LATENT_DIM = 384
ACTION_DIM = 4 # Up Down Left Right
IMG_SHAPE = (3, 64, 64)
DEFAULT_DEVICE = "cpu"

def generate_dummy_batch(batch_size, 
                         device=torch.device("cpu"), 
                         return_images=False, 
                         return_latents=True, 
                         return_actions=True,
                         return_rewards=True):
    """
    Generates random images, latents and actions for a given batch size for testing
    
    :param batch_size: Batch size
    :param device: GPU or CPU
    :param return_images: Return random images
    :param return_latens: Return random latents
    :param return_actions: Return random actions
    
    Returns:
        images : (batch_size, SEQUENCE_LENGTH, 3, 64, 64)
        latents: (batch_size, SEQUENCE_LENGTH, 384)
        actions: (batch_size, SEQUENCE_LENGTH)
    """
    
    images = None
    latents = None
    actions = None
    rewards = None
    results = []
    
    if return_images:
        images = torch.randn(batch_size, SEQUENCE_LENGTH, *IMG_SHAPE, device=DEFAULT_DEVICE)
        results.append(images)
        
    if return_latents:
        latents = torch.randn(batch_size, SEQUENCE_LENGTH, LATENT_DIM, device=DEFAULT_DEVICE)
        results.append(latents)
    
    if return_actions:
        action_id = torch.randint(0, ACTION_DIM, (batch_size, SEQUENCE_LENGTH), device=DEFAULT_DEVICE)
        actions = torch.nn.functional.one_hot(action_id, num_classes=ACTION_DIM).float()
        results.append(actions)
        
    if return_rewards:
        rewards = torch.randn(batch_size, SEQUENCE_LENGTH, 1)
        results.append(rewards)
    
    if len(results) == 1:
        return results[0] 
    return tuple(results)


def dummy_loader(num_batches, 
                 batch_size, 
                 device=torch.device("cpu"),
                 return_images=False,
                 return_latents=True,
                 return_actions=True,
                 return_rewards=True):
    """
    Loads Dummy Data Batches
    
    :param num_batches: Number of batches to produce before stopping per episode
    :param batch_size: Batch size
    :param device: GPU or CPU
    """
    for _ in range(num_batches):
        yield generate_dummy_batch(batch_size, 
                                   device,
                                   return_images=return_images,
                                   return_latents=return_latents,
                                   return_actions=return_actions,
                                   return_rewards=return_rewards)