"""验证数学/物理接口与闭环行为，避免只验证“函数能被调用”。"""
from dataclasses import replace
import numpy as np
import pytest
from mpc_tutorial.config import Config
from mpc_tutorial.controller import MPC
from mpc_tutorial.dynamics import discrete_model, prediction_matrices, rollout
from mpc_tutorial.plant import CartPlant
from mpc_tutorial.scenarios import SCENARIOS
from mpc_tutorial.simulation import simulate


@pytest.mark.parametrize("mass,substeps", [(1.0, 10), (2.4, 7), (0.7, 1)])
def test_discrete_prediction_matches_actual_mujoco(mass, substeps):
    cfg = Config(model_mass=mass, actual_mass=mass, substeps=substeps)
    plant = CartPlant(cfg, (-0.4, 0.3))
    A, B = discrete_model(cfg)
    expected = plant.state()
    for u in np.random.default_rng(12).uniform(-2, 2, 35):
        expected = A @ expected+B[:, 0]*u
        np.testing.assert_allclose(plant.step(u), expected, atol=2e-12)


def test_stacked_prediction_matches_recursive_rollout():
    cfg = Config(horizon=9)
    A, B = discrete_model(cfg)
    F, G = prediction_matrices(A, B, cfg.horizon)
    x0 = np.array([-0.6, 0.4])
    U = np.random.default_rng(3).uniform(-2, 2, cfg.horizon)
    np.testing.assert_allclose((F @ x0+G @ U).reshape(-1, 2), rollout(A, B, x0, U)[1:], atol=1e-13)


def test_unconstrained_qp_matches_independent_finite_horizon_riccati():
    # 独立的动态规划参考解，能检出 QP 的 1/2、终端权重和时间索引错误。
    cfg = Config(horizon=14, r_delta=0, force_limit=100, speed_limit=100, position_limit=100)
    mpc = MPC(cfg)
    A, B = discrete_model(cfg)
    Q = np.diag([cfg.q_position, cfg.q_velocity])
    W = cfg.terminal_scale*Q
    gains = []
    for k in range(cfg.horizon-1, -1, -1):
        K = np.linalg.solve(np.array([[cfg.r_force]])+B.T @ W @ B, B.T @ W @ A)
        gains.append(K)
        # 代码的状态代价从 x1 开始，终端 xN；x0 常数不影响最优解。
        W = Q + A.T @ W @ A-A.T @ W @ B @ K
    x = np.array([0.1, -0.02])
    expected_u = []
    for K in reversed(gains):
        u = -(K @ x).item()
        expected_u.append(u)
        x = A @ x+B[:, 0]*u
    plan = mpc.solve([0.1, -0.02], np.zeros((cfg.horizon, 2)))
    np.testing.assert_allclose(plan.controls, expected_u, atol=2e-5)


def test_soft_constraints_allow_recovery_from_outside_limit():
    cfg = Config()
    mpc = MPC(cfg)
    plan = mpc.solve([0, 2], np.zeros((cfg.horizon, 2)))
    assert plan.controls[0] < -1.99
    assert plan.slack[0, 1] > 0.5
    assert abs(plan.controls).max() <= cfg.force_limit
    np.testing.assert_array_less(abs(plan.states[1:]),
                                  np.array([cfg.position_limit, cfg.speed_limit])+plan.slack+1e-8)


def test_optimization_never_steps_real_plant():
    cfg = Config()
    plant = CartPlant(cfg)
    before = plant.state()
    mpc = MPC(cfg)
    mpc.solve(before, np.tile([1, 0], (cfg.horizon, 1)))
    np.testing.assert_array_equal(plant.state(), before)
    assert plant.data.time == 0


def test_feedback_corrects_unknown_push_while_open_loop_cannot():
    cfg = Config(horizon=40)
    scenario = SCENARIOS["push"]
    closed = simulate(cfg, scenario)
    # 同一全时长窗口的闭环和开环在第一拍应选择相同动作。
    full = replace(cfg, horizon=round(scenario.duration/cfg.dt))
    closed_full = simulate(full, scenario)
    opened = simulate(cfg, scenario, controller="openloop")
    assert closed.metrics["final_error_m"] < 0.01
    assert closed_full.metrics["final_error_m"] < 0.01
    assert opened.metrics["final_error_m"] > 1.0
    np.testing.assert_allclose(opened.actions, opened.plans[0]["controls"], atol=1e-12)
    assert len(opened.plans[-1]["controls"]) == 1
    assert abs(closed_full.actions[0]-opened.actions[0]) < 1e-5
    assert closed.metrics["max_abs_force_N"] <= cfg.force_limit
    assert all(p["status"] == "solved" for p in closed.plans)


def test_mismatched_model_is_visible_in_next_state_error():
    cfg = Config(model_mass=1, actual_mass=2)
    plant = CartPlant(cfg, (0, 0))
    A, B = discrete_model(cfg)
    predicted = A @ plant.state()+B[:, 0]*1.5
    actual = plant.step(1.5)
    assert predicted[1] == pytest.approx(2*actual[1])


def test_braking_begins_before_target():
    cfg = Config()
    sc = SCENARIOS["brake"]
    plan = MPC(cfg).solve(sc.x0, sc.reference(0, cfg.dt, cfg.horizon))
    assert sc.x0[0] < sc.initial_goal
    assert plan.controls[0] < -0.1


@pytest.mark.parametrize("kwargs", [{"horizon":0}, {"actual_mass":-1}, {"physics_dt":float('nan')}])
def test_invalid_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
