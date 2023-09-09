from psp.pipeline.gbdt import GBDTPipeline
from psp.pipeline.rnn import RNNPipeline

pipelines = {
    "gbdt": GBDTPipeline,
    "rnn": RNNPipeline,
}
