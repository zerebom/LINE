import hydra
from omegaconf import DictConfig


@hydra.main(config_path='config.yaml')
def myapp(cfg: DictConfig) -> None:
    print(cfg.model)


myapp()
