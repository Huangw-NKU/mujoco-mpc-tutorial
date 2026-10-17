"""将真实实验日志变成离线可打开的 HTML 和概览图，不依赖 CDN。"""
from dataclasses import asdict
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_report(runs, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    template = (Path(__file__).parent / "report_template.html").read_text(encoding="utf-8")
    payload = json.dumps([asdict(r) for r in runs], ensure_ascii=False).replace("</", "<\\/")
    path = directory / "report.html"
    path.write_text(template.replace("__RUN_DATA__", payload), encoding="utf-8")
    (directory / "metrics.json").write_text(
        json.dumps({r.label: r.metrics for r in runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    for i, run in enumerate(runs):
        run.save(directory / f"run_{i}_{run.controller}.json")
    plot_overview(runs, directory / "overview.png")
    return path


def plot_overview(runs, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, layout="constrained")
    colors = ["#247ac4", "#c16d21", "#8570a8", "#467f4e"]
    for i, r in enumerate(runs):
        t, x = np.array(r.time), np.array(r.states)
        color = colors[i % len(colors)]
        axes[0].plot(t, x[:, 0], label=r.label, color=color)
        axes[1].plot(t, x[:, 1], color=color)
        axes[2].step(t[:-1], r.actions, where="post", color=color)
    r = runs[0]
    axes[0].step(r.time, r.goals, where="post", color="black", linestyle="--", label="Target")
    for sign in [-1, 1]:
        axes[0].axhline(sign*r.config["position_limit"], color="0.6", ls=":")
        axes[1].axhline(sign*r.config["speed_limit"], color="0.6", ls=":")
        axes[2].axhline(sign*r.config["force_limit"], color="0.6", ls=":")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(r.time[0], r.time[-1])
    axes[0].legend(ncol=3, fontsize=9)
    axes[0].set_ylabel("Position (m)")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[2].set_ylabel("Motor force (N)")
    axes[2].set_xlabel("Simulation time (s)")
    fig.suptitle(f"MuJoCo experiment: {r.scenario} | dt={r.config['physics_dt']*r.config['substeps']:.3f} s")
    fig.savefig(path, dpi=140)
    plt.close(fig)
