# GDP 2: Latent World Model Proof-of-Concept

## The Interface Contract (The Handshake)
All members must adhere to these shapes to ensure integration.

* **Latent Dimension:** Fixed at `384` (Matches DINOv2 ViT-S/14 output)
* **Sequence Length (T):** Fixed at `16` steps
* **Input Image Size:** Fixed at `(64, 64, 3)` RGB
* **Batch Structure:** Tensors passed between Buffer and Model must be `(Batch Size, 16, 384)`

## Folder Structure
Please place your files exactly as defined in the Execution Plan PDF.
- Member A: `src/env/`
- Member B: `src/models/transformer.py`
- Member C: `src/models/encoder.py` and `src/data/`
- Member D: `src/utils/`
