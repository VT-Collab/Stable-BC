
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict
import os, sys
from get_demos import get_dataset
from process_demos import process_demos
from train import train_model
from rollout import rollout_policy


@hydra.main(version_base="1.2", config_name='config', config_path='./cfg')
def main(cfg=DictConfig):
    if not os.path.exists('data/user_{}/{}_dp/'.format(cfg.user, cfg.num_dp)):
        os.makedirs('data/user_{}/{}_dp'.format(cfg.user, cfg.num_dp))

    if cfg.get_demo:
        get_dataset(cfg)
        process_demos(cfg)

    if cfg.train:
        train_model(cfg)

    if cfg.rollout:
        rollout_policy(cfg)
    

if __name__ == "__main__":
    main()
