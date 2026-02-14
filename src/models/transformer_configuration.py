import torch

class TransformerWMConfiguration:
    """
    World Model Transformer Params
    4 Actions
    4 Layers
    4 Attention Heads
    16 Sequence Length
    384 Latent Dimension from DINOv2 ViT-S/16
    4x MLP Ratio
    1e-4 Learning Rate
    """
    LATENT_DIM = 384
    SEQUENCE_LENGTH = 16
    NUM_LAYERS = 4
    NUM_HEADS = 4
    ACTION_DIM = 4
    MLP_RATIO = 4
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")