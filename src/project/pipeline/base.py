import logging

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Pipeline:
    pipeline_name = "base"

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def create_dataset(self):
        ...

    def train(self):
        ...

    def cv(self):
        ...

    def tune(self):
        ...

    def run(self):
        self.create_dataset()

        task = getattr(self, self.cfg.task)
        if task is None:
            raise NotImplementedError(f"Task {self.cfg.task} not found.")

        logger.info(f"Running task {self.cfg.task}")
        task()
