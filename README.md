
# PyTorch Chatbot Tutorial (Low-End Device Optimized)

## Changes from Original
- Default hidden size reduced to 256
- Fewer GRU layers (1 by default)
- Gradient clipping reduced for stability
- Simplified attention computation
- Default batch size recommendation: 16–32

## Quick Start
1. Install dependencies:
    ```bash
    pip install torch torchvision
    ```
2. Place your dataset under `data/`.
3. Train:
    ```bash
    python train.py --batch_size 16 --hidden_size 256 --n_layers 1
    ```
4. Chat:
    ```bash
    python chat.py
    ```
