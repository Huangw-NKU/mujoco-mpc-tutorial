"""先改这里：所有参数都有单位，预测模型质量与仿真质量可以故意不同。"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    horizon: int = 40              # N：看未来 40 * 0.05 = 2 秒
    physics_dt: float = 0.005      # h，s
    substeps: int = 10             # 每个控制周期内保持同一推力
    model_mass: float = 1.0        # 预测器相信的质量，kg
    actual_mass: float = 1.0       # MuJoCo 中的小车质量，kg
    force_limit: float = 2.0       # N，硬约束
    position_limit: float = 1.8    # m，软约束
    speed_limit: float = 1.3       # m/s，软约束
    q_position: float = 20.0       # 更大：更在意位置误差
    q_velocity: float = 1.0        # 更大：更在意停车/减速
    r_force: float = 0.08          # 更大：更节省推力
    r_delta: float = 0.03          # 更大：更平滑，惩罚相邻推力之差
    terminal_scale: float = 3.0    # Qf = terminal_scale * Q；不是稳定性证明
    slack_quadratic: float = 1000.0
    slack_linear: float = 100.0

    @property
    def dt(self) -> float:
        return self.physics_dt * self.substeps

    def __post_init__(self):
        if self.horizon < 1 or not isinstance(self.horizon, int):
            raise ValueError("horizon 必须是正整数")
        if self.substeps < 1 or not isinstance(self.substeps, int):
            raise ValueError("substeps 必须是正整数")
        positive = (self.physics_dt, self.model_mass, self.actual_mass,
                    self.force_limit, self.position_limit, self.speed_limit,
                    self.r_force, self.slack_quadratic)
        nonnegative = (self.q_position, self.q_velocity, self.r_delta,
                       self.terminal_scale, self.slack_linear)
        import math
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("周期、质量、限制、r_force 和 slack_quadratic 必须为有限正数")
        if not all(math.isfinite(v) and v >= 0 for v in nonnegative):
            raise ValueError("状态权重、终端权重和其余惩罚项必须为有限非负数")
