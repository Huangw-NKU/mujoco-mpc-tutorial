"""【4 滚动】教学主循环：测量 → 预测/优化 → 只执行第一项 → 再测量。"""
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import numpy as np
from .config import Config
from .controller import MPC, Plan
from .dynamics import discrete_model, rollout
from .plant import CartPlant
from .scenarios import Scenario


@dataclass
class Run:
    label: str
    controller: str
    scenario: str
    config: dict
    time: list
    states: list                  # K+1 个真实状态，包含最终状态
    goals: list                   # K+1 个当时可见目标
    actions: list                 # K 个实际执行动作
    disturbances: list
    plans: list                   # 每拍的 {states,controls,reference,slack,solve_ms,status}
    metrics: dict

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def simulate(cfg: Config, scenario: Scenario, controller="mpc", preview=False,
             duration=None, label=None, on_frame=None, live_push=None):
    duration = scenario.duration if duration is None else duration
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration 必须为有限正数")
    K = int(np.ceil(duration/cfg.dt))
    plant = CartPlant(cfg, scenario.x0)
    # 开环在 t=0 规划整个实验时长，随后不再求解；不会把“没有后续计划”冒充开环。
    plan_cfg = replace(cfg, horizon=K) if controller == "openloop" else cfg
    mpc = MPC(plan_cfg) if controller in ("mpc", "openloop") else None
    A, B = discrete_model(plan_cfg)
    cached = None
    states, goals = [plant.state().tolist()], [scenario.goal(0)]
    times, actions, forces, plans = [0.0], [], [], []
    for k in range(K):
        t = k*cfg.dt
        # STEP 1 测量：从仿真读当前真实位置和速度（现实中需传感器/状态估计器）。
        x_now = plant.state()
        N = plan_cfg.horizon
        reference = scenario.reference(t, cfg.dt, N, preview)
        # STEP 2-3 预测和优化：求未来整段动作，不接触真实仿真 data。
        if controller == "mpc":
            plan = mpc.solve(x_now, reference)
        elif controller == "openloop":
            if cached is None:
                cached = mpc.solve(x_now, reference)
            controls = cached.controls[k:].copy()
            predicted = rollout(A, B, x_now, controls)
            slack = np.maximum(abs(predicted[1:])-[cfg.position_limit, cfg.speed_limit], 0)
            plan = Plan(controls, predicted, slack, cached.solve_ms if k == 0 else 0.0,
                        "solved once" if k == 0 else "not replanned")
            # 图中参考也要保留 t=0 制定计划时的信息；未来目标变了并不修改控制。
            reference = scenario.reference(0, cfg.dt, N, preview)
            reference = reference[k:]
        elif controller == "pd":
            u = float(np.clip(4.0*(scenario.goal(t)-x_now[0])-1.2*x_now[1],
                              -cfg.force_limit, cfg.force_limit))
            predicted = rollout(A, B, x_now, [u])
            needed = np.maximum(abs(predicted[1:])-[cfg.position_limit, cfg.speed_limit], 0)
            plan = Plan(np.array([u]), predicted, needed, 0.0, "PD (no optimization)")
            reference = reference[:1]
        else:
            raise ValueError(f"未知 controller: {controller}")
        # STEP 4 只取第一项。预测中后面的动作是计划，下次允许全部改变。
        u = float(plan.controls[0])
        force = scenario.force(t) + (live_push(t) if live_push else 0.0)
        plant.set_goal(scenario.goal(t))
        frame = dict(time=t, state=x_now.tolist(), goal=scenario.goal(t), action=u,
                     force=force, plan=plan, reference=reference)
        if on_frame is not None and on_frame(frame, plant) is False:
            break
        actions.append(u)
        forces.append(force)
        plans.append(dict(states=plan.states.tolist(), controls=plan.controls.tolist(),
                          reference=reference.tolist(), slack=plan.slack.tolist(),
                          solve_ms=plan.solve_ms, status=plan.status))
        # STEP 5 在外力存在的物理世界中运行一个控制周期；之后回到 STEP 1。
        x_next = plant.step(u, force)
        if mpc is not None:
            mpc.record_applied(u)
        states.append(x_next.tolist())
        times.append((k+1)*cfg.dt)
        goals.append(scenario.goal((k+1)*cfg.dt))
    x = np.asarray(states)
    error = x[:, 0]-np.asarray(goals)
    solve_ms = np.array([p["solve_ms"] for p in plans])
    optimized = solve_ms[solve_ms > 0]
    metrics = dict(
        position_rmse_m=float(np.sqrt(np.mean(error**2))),
        final_error_m=float(abs(error[-1])), final_speed_m_s=float(abs(x[-1, 1])),
        max_speed_m_s=float(abs(x[:, 1]).max()),
        max_position_violation_m=float(np.maximum(abs(x[:, 0])-cfg.position_limit, 0).max()),
        max_speed_violation_m_s=float(np.maximum(abs(x[:, 1])-cfg.speed_limit, 0).max()),
        max_abs_force_N=float(max(map(abs, actions), default=0)),
        control_energy_N2s=float(np.sum(np.square(actions))*cfg.dt),
        mean_plan_ms=float(optimized.mean()) if len(optimized) else 0.0,
        max_plan_ms=float(optimized.max()) if len(optimized) else 0.0,
        p95_plan_ms=float(np.percentile(optimized, 95)) if len(optimized) else 0.0,
        planned_steps=len(plans),
    )
    return Run(label or controller, controller, scenario.name, asdict(plan_cfg), times,
               states, goals, actions, forces, plans, metrics)
