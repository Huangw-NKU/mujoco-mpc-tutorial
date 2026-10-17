"""【2 优化】用 OSQP 解凸二次规划。求解器解数学问题；MuJoCo 推进物理世界。"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import scipy.sparse as sp
import osqp
from .config import Config
from .dynamics import discrete_model, prediction_matrices


@dataclass
class Plan:
    controls: np.ndarray           # 长度 N：整段计划，闭环仅执行 controls[0]
    states: np.ndarray             # (N+1,2)，包含当前状态
    slack: np.ndarray              # (N,2)，位置/速度界限允许放宽的量
    solve_ms: float
    status: str


class MPC:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        N = cfg.horizon
        self.A, self.B = discrete_model(cfg)
        self.F, self.G = prediction_matrices(self.A, self.B, N)
        # J 的状态部分：x1...xN-1 用 Q，末端 xN 用 Qf。
        weights = np.tile([cfg.q_position, cfg.q_velocity], N)
        weights[-2:] *= cfg.terminal_scale
        self.Qbar = np.diag(weights)
        # D @ U - e0*u_previous = [u0-u_previous, u1-u0, ...]。
        self.D = np.eye(N) - np.eye(N, k=-1)
        H = (self.G.T @ self.Qbar @ self.G + cfg.r_force*np.eye(N)
             + cfg.r_delta*self.D.T @ self.D)
        # OSQP 用 1/2 z'Pz + q'z，所以这里的 Hessian 要乘 2。
        # 决策变量 z=[U,S]，S 与未来状态逐元素对应，S>=0。
        P = sp.block_diag((2*H, 2*cfg.slack_quadratic*sp.eye(2*N)), format="csc")
        Iu, Is = sp.eye(N), sp.eye(2*N)
        Zu = sp.csc_matrix((N, 2*N))
        Zs = sp.csc_matrix((2*N, N))
        # 四组约束：推力；X-S<=upper；-X-S<=-lower；S>=0。
        C = sp.vstack([
            sp.hstack([Iu, Zu]),
            sp.hstack([sp.csc_matrix(self.G), -Is]),
            sp.hstack([-sp.csc_matrix(self.G), -Is]),
            sp.hstack([Zs, Is]),
        ], format="csc")
        self.state_upper = np.tile([cfg.position_limit, cfg.speed_limit], N)
        self.lower = np.r_[np.full(N, -cfg.force_limit),
                           np.full(4*N, -np.inf), np.zeros(2*N)]
        self.upper = np.r_[np.full(N, cfg.force_limit),
                           self.state_upper, self.state_upper, np.full(2*N, np.inf)]
        self.solver = osqp.OSQP()
        self.solver.setup(P=sp.triu(P, format="csc"), q=np.zeros(3*N), A=C,
                          l=self.lower, u=self.upper, verbose=False,
                          eps_abs=1e-5, eps_rel=1e-5, max_iter=20000,
                          polishing=True, warm_starting=True)
        self.previous_u = 0.0
        self.previous_plan = None

    def solve(self, x_now, reference):
        """reference 为 (N,2)，每个未来节点都有位置/速度目标。"""
        start = perf_counter()
        N, cfg = self.cfg.horizon, self.cfg
        x_now = np.asarray(x_now, dtype=float)
        reference = np.asarray(reference, dtype=float)
        if x_now.shape != (2,) or reference.shape != (N, 2):
            raise ValueError(f"x_now 需 (2,)，reference 需 ({N},2)")
        if not np.isfinite(x_now).all() or not np.isfinite(reference).all():
            raise ValueError("状态和参考值必须有限")
        free = self.F @ x_now
        error = free - reference.ravel()
        previous = np.zeros(N)
        previous[0] = self.previous_u
        q_u = 2*(self.G.T @ self.Qbar @ error - cfg.r_delta*self.D.T @ previous)
        q = np.r_[q_u, np.full(2*N, cfg.slack_linear)]
        upper = self.upper.copy()
        upper[N:3*N] = self.state_upper - free
        upper[3*N:5*N] = self.state_upper + free
        # 只有 q 和约束上界随当前状态/目标变化；P、C 预先构造一次。
        self.solver.update(q=q, u=upper)
        if self.previous_plan is not None:
            # 上次剩下的计划左移一个控制周期，尾部重复；只是初值，仍会重新优化。
            shifted = np.r_[self.previous_plan.controls[1:], self.previous_plan.controls[-1]]
            slack = np.maximum(abs(self.F @ x_now + self.G @ shifted) - self.state_upper, 0)
            self.solver.warm_start(x=np.r_[shifted, slack])
        result = self.solver.solve(raise_error=False)
        if result.info.status != "solved" or result.x is None:
            # 教学工程选择明确失败；真实机器人需要独立设计可验证的后备控制。
            raise RuntimeError(f"QP 求解失败：{result.info.status}; 当前状态={x_now}")
        raw_u = result.x[:N]
        if np.max(abs(raw_u)) > cfg.force_limit + 1e-3:
            raise RuntimeError("QP 结果超出推力容差，拒绝执行")
        controls = np.clip(raw_u, -cfg.force_limit, cfg.force_limit)
        predicted = (free + self.G @ controls).reshape(N, 2)
        # 显示实际数值计划所需的松弛，包含最后的微小裁剪修正。
        slack = np.maximum(abs(predicted) - self.state_upper.reshape(N, 2), 0)
        plan = Plan(controls, np.vstack([x_now, predicted]), slack,
                    1000*(perf_counter()-start), result.info.status)
        self.previous_plan = plan
        return plan

    def record_applied(self, u):
        """平滑惩罚基于上一个真正执行的 u，而不是某个未来计划值。"""
        self.previous_u = float(u)
