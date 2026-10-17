"""入口：run 单次实验，compare 对照，inspect 拆解某一次优化。"""
import argparse
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .config import Config
from .controller import MPC
from .report import write_report
from .scenarios import SCENARIOS
from .simulation import simulate


def make_parser():
    parser = argparse.ArgumentParser(description="MuJoCo MPC 中文教学项目")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [("run", "运行单次实验"), ("compare", "运行控制器/窗口/质量/预瞄对照")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--scenario", choices=SCENARIOS, default="demo" if name == "run" else "push")
        p.add_argument("--duration", type=float)
        p.add_argument("--horizon", type=int, default=40)
        p.add_argument("--mass", type=float, default=1, help="MuJoCo 真实质量 (kg)")
        p.add_argument("--model-mass", type=float, default=1, help="预测器质量 (kg)")
        p.add_argument("--speed-limit", type=float, default=1.3)
        p.add_argument("--out", type=Path, default=Path("outputs")/name)
        if name == "run":
            p.add_argument("--controller", choices=["mpc", "pd", "openloop"], default="mpc")
            p.add_argument("--viewer", action="store_true", help="打开 MuJoCo 窗口；P 手动加外力，空格暂停")
            p.add_argument("--video", type=Path, help="导出 mp4；无桌面时用 MUJOCO_GL=egl")
            p.add_argument("--preview", action="store_true", help="告知控制器未来的目标切换（不预告外力）")
        else:
            p.add_argument("--kind", choices=["controllers", "horizon", "mass", "preview"], default="controllers")
    p = sub.add_parser("inspect", help="打印一次预测与优化，适合调试器单步跟读")
    p.add_argument("--position", type=float, default=-0.5)
    p.add_argument("--velocity", type=float, default=1.3)
    p.add_argument("--target", type=float, default=0.0)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--out", type=Path, default=Path("outputs/inspect/plan.json"))
    return parser


def summary(run):
    m = run.metrics
    print(f"{run.label}: RMSE={m['position_rmse_m']:.4f} m, "
          f"末端误差={m['final_error_m']:.4f} m, "
          f"最高速度={m['max_speed_m_s']:.3f} m/s, "
          f"规划 P95={m['p95_plan_ms']:.2f} ms")


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.command == "inspect":
        cfg = Config(horizon=args.horizon)
        mpc = MPC(cfg)
        x = np.array([args.position, args.velocity])
        ref = np.tile([args.target, 0.0], (cfg.horizon, 1))
        plan = mpc.solve(x, ref)
        print("x = [位置(m), 速度(m/s)]；u 是推力(N)")
        print("A =\n", mpc.A, "\nB =\n", mpc.B)
        print(f"F.shape={mpc.F.shape}, G.shape={mpc.G.shape}")
        print(f"当前 x={x}；目标=[{args.target}, 0]；窗口={cfg.horizon*cfg.dt:.2f} s")
        print(" k |     u_k (N) |  p_(k+1) (m) | v_(k+1) (m/s)")
        for k in range(min(12, cfg.horizon)):
            print(f"{k:2d} | {plan.controls[k]:11.5f} | {plan.states[k+1,0]:12.5f} | {plan.states[k+1,1]:13.5f}")
        print(f"末端状态={plan.states[-1]}；求解={plan.status}，耗时={plan.solve_ms:.3f} ms")
        print(f"本拍只执行 u[0]={plan.controls[0]:.5f} N；下一拍用新状态重算。")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(x=x.tolist(), A=mpc.A.tolist(), B=mpc.B.tolist(),
            F=mpc.F.tolist(), G=mpc.G.tolist(), reference=ref.tolist(),
            controls=plan.controls.tolist(), states=plan.states.tolist(), slack=plan.slack.tolist()),
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"完整矩阵与计划：{args.out.resolve()}")
        return 0
    cfg = Config(horizon=args.horizon, actual_mass=args.mass, model_mass=args.model_mass,
                 speed_limit=args.speed_limit)
    scenario = SCENARIOS[args.scenario]
    runs = []
    if args.command == "run":
        from .visualization import LiveViewer, Video
        live = LiveViewer(cfg) if args.viewer else None
        video = Video(args.video, cfg) if args.video else None
        def on_frame(frame, plant):
            if live is not None and live.frame(frame, plant) is False:
                return False
            if video is not None:
                video.frame(frame, plant)
            return True
        if live:
            print("MuJoCo 窗口：P 添加 4 N、持续 0.2 s 的外力；空格暂停/继续；关闭窗口结束。")
        try:
            runs.append(simulate(cfg, scenario, args.controller, args.preview, args.duration,
                                 label=(f"MPC N={cfg.horizon}" if args.controller == "mpc" else
                                        "PD kp=4 kd=1.2" if args.controller == "pd" else
                                        "Open loop (full-run plan)"),
                                 on_frame=on_frame, live_push=live.force if live else None))
        finally:
            if video:
                video.close()
            if live:
                live.close()
    else:
        cases = []
        if args.kind == "controllers":
            # 给开环与该 MPC 完全相同的全时长窗口，隔离“是否反馈”的影响。
            duration = scenario.duration if args.duration is None else args.duration
            full = max(1, int(np.ceil(duration/cfg.dt)))
            cases = [(cfg, "mpc", False, f"MPC N={cfg.horizon}"),
                     (replace(cfg, horizon=full), "mpc", False, "MPC full horizon"),
                     (cfg, "openloop", False, "Open loop (same full horizon)"),
                     (cfg, "pd", False, "PD kp=4 kd=1.2")]
        elif args.kind == "horizon":
            cases = [(replace(cfg, horizon=n), "mpc", False, f"MPC N={n}")
                     for n in sorted(set([4, 12, args.horizon]))]
        elif args.kind == "mass":
            cases = [(cfg, "mpc", False, "MPC nominal plant"),
                     (replace(cfg, actual_mass=2*cfg.actual_mass), "mpc", False, "MPC heavier, wrong model"),
                     (replace(cfg, actual_mass=2*cfg.actual_mass, model_mass=2*cfg.actual_mass), "mpc", False, "MPC heavier, matched model")]
        else:
            if not np.isfinite(scenario.switch_time):
                raise ValueError("preview 对照请用 --scenario switch 或 demo")
            cases = [(cfg, "mpc", False, "MPC current target only"),
                     (cfg, "mpc", True, "MPC future target preview")]
        for config, controller, preview, label in cases:
            run = simulate(config, scenario, controller, preview, args.duration, label)
            summary(run)
            runs.append(run)
    if args.command == "run":
        summary(runs[0])
    report = write_report(runs, args.out)
    print(f"离线交互报告：{report.resolve()}")
    print(f"概览图：{(args.out/'overview.png').resolve()}")
    return 0
