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
        tasks = self.cfg.tasks

        if isinstance(tasks, str):
            tasks = [tasks]

        for task in tasks:
            task_fn = getattr(self, task)
            if task is None:
                raise NotImplementedError(f"Task task not found.")

            logger.info(f"Running task <{task}>")
            task_fn()
