import os
import hydra

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from run_train import run_multiprocess

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    world_size = 1
    port = 29500
    run_multiprocess(0, world_size, cfg, port)

if __name__ == "__main__":
    main()