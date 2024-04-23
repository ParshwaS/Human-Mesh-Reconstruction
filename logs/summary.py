from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def summarize_log(logfile: Path):
    event_order = []
    with logfile.open() as f:
        for line in f:
            event, time = line.strip().split(": ")
            if event not in event_order:
                event_order.append(event)
            else:
                break

    data = (
        pd.read_table(logfile, sep=":", header=None, names=["event", "time"])
        .groupby("event")["time"]
        .apply(np.array)
    )[event_order]

    return data


def plot_log(data: pd.Series, logfile: Path):
    fig = plt.figure(figsize=(8, 5))
    plt.plot(data.index, data.map(lambda x: x.mean()), color="tab:blue", label="mean")
    plt.fill_between(
        data.index,
        data.map(lambda x: x.mean() - x.std()),
        data.map(lambda x: x.mean() + x.std()),
        alpha=0.3,
        color="tab:blue",
        label="std",
    )
    plt.fill_between(
        data.index,
        data.map(lambda x: x.min()),
        data.map(lambda x: x.max()),
        alpha=0.15,
        color="tab:blue",
        label="range",
    )
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(logfile.with_suffix(".png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("logfile", type=Path)
    args = parser.parse_args()
    logfile = Path(args.logfile)

    plot_log(summarize_log(logfile), logfile)
