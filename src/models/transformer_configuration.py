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

    ─────────────────────────────────────────────────────────────
    MODIFICATION — Member E (HP Sweep, CDR Sprint)
    ─────────────────────────────────────────────────────────────
    Added classmethod `from_params` so the hyperparameter sweep
    can instantiate a config with explicit values without touching
    the default constants used by the rest of the codebase.

    All existing code that does:
        config = TransformerWMConfiguration()
        model  = DinoWorldModel(config)
    is completely unaffected.
    ─────────────────────────────────────────────────────────────
    """

    # ── Fixed by interface contract — DO NOT CHANGE ───────────
    LATENT_DIM      = 384   # DINOv2 ViT-S/14 CLS token dim
    ACTION_DIM      = 4     # N / S / E / W

    # ── Free parameters (current best-known defaults) ─────────
    SEQUENCE_LENGTH = 24
    NUM_LAYERS      = 2
    NUM_HEADS       = 8
    MLP_RATIO       = 2
    LEARNING_RATE   = 3e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_params(
        cls,
        num_heads:       int   = 4,
        num_layers:      int   = 4,
        mlp_ratio:       int   = 4,
        learning_rate:   float = 1e-4,
        sequence_length: int   = 16,
    ) -> "TransformerWMConfiguration":
        """
        Create a config instance with explicit hyperparameter values.

        Used by the Optuna sweep (src/tuning/train_sweep.py) so that
        each trial can instantiate a fresh model with different params
        without modifying the class-level defaults.

        Enforces the interface contract:
            LATENT_DIM % num_heads == 0  (head_dim must be integer)

        Args:
            num_heads:       Number of attention heads.
                             Valid values: any divisor of 384
                             (e.g. 2→192-dim, 4→96-dim, 6→64-dim, 8→48-dim)
            num_layers:      Number of transformer blocks.
            mlp_ratio:       MLP expansion ratio inside each block.
            learning_rate:   Adam learning rate.
            sequence_length: Context window length (steps).

        Returns:
            A TransformerWMConfiguration instance with overridden params.

        Raises:
            ValueError: If LATENT_DIM % num_heads != 0.
        """
        if 384 % num_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} does not divide LATENT_DIM=384. "
                f"head_dim would be non-integer. "
                f"Valid values: {[h for h in [1,2,3,4,6,8,12,16,24,32,48,96,128,192,384] if 384 % h == 0]}"
            )

        cfg                  = cls()
        cfg.NUM_HEADS        = num_heads
        cfg.NUM_LAYERS       = num_layers
        cfg.MLP_RATIO        = mlp_ratio
        cfg.LEARNING_RATE    = learning_rate
        cfg.SEQUENCE_LENGTH  = sequence_length
        return cfg