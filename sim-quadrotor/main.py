
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict
import os, sys
from get_demos import get_dataset
from train import train_model
from rollout import rollout_policy


@hydra.main(version_base="1.2", config_name='config', config_path='./cfg')
def main(cfg=DictConfig):
    if not os.path.exists('data/'):
        os.makedirs('data/')

    if cfg.get_demo:
        get_dataset(cfg)

    if cfg.train:
        train_model(cfg)

    if cfg.rollout:
        rollout_policy(cfg)
    

if __name__ == "__main__":
    main()
