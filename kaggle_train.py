import os
import sys
import hydra
import numpy as np

# Set up environment for Kaggle
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from run_train import run_multiprocess

@hydra.main(version_base=None, config_path="configs", config_name="config_kaggle")
def main(cfg):
    """Main entry point for Kaggle training.
    
    This script is optimized for Kaggle's GPU and memory constraints.
    It uses:
    - Smaller batch size (64 instead of 512)
    - Gradient accumulation for effective larger batches
    - Single GPU training
    - Reduced number of iterations for testing
    """
    world_size = 1
    port = int(np.random.randint(10000, 20000))
    
    print(f"Starting Kaggle training with config: {cfg.training}")
    print(f"Using GPU: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    run_multiprocess(0, world_size, cfg, port)

if __name__ == "__main__":
    import torch
    main()