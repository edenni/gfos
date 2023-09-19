from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def plot_univar_dist(
    data: Union[pd.Series, np.ndarray],
    feature: str,
    bins: int = 250,
    ax: Optional[Axes] = None,
) -> None:
    """Plot univariate distribution.

    Parameters:
        data: univariate data to plot
        feature: feature name of the data
        bins: number of bins
        ax: user-specified axes

    Return:
        None
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=data, bins=bins, kde=True, ax=ax)
    ax.axvline(
        x=data.mean(),
        color="orange",
        linestyle="dotted",
        linewidth=1.5,
        label="Mean",
    )
    ax.axvline(
        x=data.median(),
        color="green",
        linestyle="dotted",
        linewidth=1.5,
        label="Median",
    )
    ax.axvline(
        x=data.mode().values[0],
        color="red",
        linestyle="dotted",
        linewidth=1.5,
        label="Mode",
    )
    ax.set_title(
        f"{feature.upper()} Distibution\n"
        f"Min {round(data.min(), 2)} | "
        f"Max {round(data.max(), 2)} | "
        f"Skewness {round(data.skew(), 2)} | "
        f"Kurtosis {round(data.kurtosis(), 2)}"
    )
    ax.set_xlabel(f"{feature}")
    ax.set_ylabel("Bin Count")
    ax.legend()
    if ax is None:
        plt.show()
