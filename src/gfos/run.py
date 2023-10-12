import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from gfos.pipeline import pipelines
from gfos.utils.rich import print_config_tree

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="run")
def main(cfg: DictConfig):
    if cfg.pipeline not in pipelines:
        raise NotImplementedError(
            f"Pipeline {cfg.pipeline} not found.\n"
            + f"Avaliable pipelines are: {list(pipelines.keys())}"
        )

    if cfg.ignore_warnings:
        import warnings

        warnings.filterwarnings("ignore")

    print_config_tree(cfg, save_to_file=False)

    pl.seed_everything(cfg.get("seed", 42))

    pipeline = pipelines[cfg.pipeline]
    logger.info(f"Running pipeline {pipeline}.")
    pipeline(cfg).run()


if __name__ == "__main__":
    main()
