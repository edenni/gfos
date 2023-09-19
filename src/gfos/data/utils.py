import os
from collections import defaultdict


def load_layout(
    base_dir: str, compile_type: str, model_type: str | None = None
):
    if model_type is not None:
        assert model_type in (
            "nlp",
            "xla",
        ), f"model_type must be nlp or xla but got {model_type}"

    assert compile_type in (
        "default",
        "random",
    ), f"compile_type must be default or random but got {compile_type}"

    dfs = defaultdict(list)

    if model_type is None:
        model_types = ("nlp", "xla")
    else:
        model_types = (model_type,)

    dirs = [
        os.path.join(base_dir, model_type, compile_type, training)
        for model_type in model_types
        for training in ["train", "valid", "test"]
    ]

    for path in dirs:
        split = path.split("\\")[-1]
        files = os.listdir(path)

        dfs[split] += [os.path.join(path, file) for file in files]

    return dfs
