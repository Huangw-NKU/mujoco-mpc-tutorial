"""【1 预测】把一步动力学叠成整个预测窗口，纯 NumPy，无 MuJoCo 状态读写。"""
import numpy as np
from .config import Config


def discrete_model(cfg: Config):
    """与无阻尼 MuJoCo Euler 的 n 个子步一致，不混用连续时间精确离散。"""
    h, n, m = cfg.physics_dt, cfg.substeps, cfg.model_mass
    # 单子步：v+=h*u/m; p+=h*v（使用更新后的速度，半隐式 Euler）。
    A = np.array([[1.0, cfg.dt], [0.0, 1.0]])
    B = np.array([[h*h*n*(n+1)/(2*m)], [cfg.dt/m]])
    return A, B


def prediction_matrices(A, B, horizon):
    """X = F @ x_now + G @ U；X 按 [p1,v1,p2,v2,...,pN,vN] 排列。"""
    nx, nu = B.shape
    F = np.zeros((nx*horizon, nx))
    G = np.zeros((nx*horizon, nu*horizon))
    for i in range(horizon):
        F[nx*i:nx*(i+1)] = np.linalg.matrix_power(A, i+1)
        for j in range(i+1):
            G[nx*i:nx*(i+1), nu*j:nu*(j+1)] = np.linalg.matrix_power(A, i-j) @ B
    return F, G


def rollout(A, B, x0, controls):
    """逐步预测，用于教学和独立验证 F/G；第 0 行就是当前状态。"""
    states = [np.asarray(x0, dtype=float).copy()]
    for u in controls:
        states.append(A @ states[-1] + B[:, 0]*u)
    return np.asarray(states)
