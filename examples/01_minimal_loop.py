"""先读这 30 行，再读工程中的日志、图形和 CLI。运行：uv run python examples/01_minimal_loop.py"""
from mpc_tutorial.config import Config
from mpc_tutorial.controller import MPC
from mpc_tutorial.plant import CartPlant
from mpc_tutorial.scenarios import SCENARIOS

cfg = Config(horizon=40)
scenario = SCENARIOS["push"]
plant = CartPlant(cfg, scenario.x0)
mpc = MPC(cfg)

for k in range(round(scenario.duration/cfg.dt)):
    t = k*cfg.dt
    # 1. 测量：必须读真实世界的新状态，不能把上次的预测当成事实。
    x_now = plant.state()
    # 2. 定义未来目标；默认未来 40 步保持当前目标，不知道之后的外力。
    reference = scenario.reference(t, cfg.dt, cfg.horizon)
    # 3. 预测 + 优化：在不推动真实仿真的前提下求出 40 个推力。
    plan = mpc.solve(x_now, reference)
    # 4. 只执行第一项，保持一个控制周期 0.05 s。
    u = plan.controls[0]
    plant.step(u, external_force=scenario.force(t))
    mpc.record_applied(u)
    # 5. 下一次循环重新读状态并求解，剩下 39 项仅用于热启动。
    if k % 10 == 0:
        print(f"t={t:.2f}s x={x_now.round(3)} u={u:+.3f}N predicted_next={plan.states[1].round(3)} actual_next={plant.state().round(3)}")
