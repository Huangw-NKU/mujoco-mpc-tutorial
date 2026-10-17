"""所有控制器经历相同的初始条件、目标和外力；不偷看未来扰动。"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    x0: tuple
    duration: float
    initial_goal: float
    switch_time: float = float("inf")
    next_goal: float = -1.0
    push_start: float = float("inf")
    push_duration: float = 0.2
    push_force: float = 4.0

    def goal(self, t):
        return self.initial_goal if t < self.switch_time - 1e-9 else self.next_goal

    def force(self, t):
        return self.push_force if self.push_start-1e-9 <= t < self.push_start+self.push_duration-1e-9 else 0.0

    def reference(self, t, dt, N, preview=False):
        # 默认只知道“此刻的目标”，未来 N 步都用它；preview 才知道预定的目标切换。
        goals = [self.goal(t+(i+1)*dt) if preview else self.goal(t) for i in range(N)]
        return np.column_stack([goals, np.zeros(N)])


SCENARIOS = {
    "reach": Scenario("reach", "从静止到达目标", (-1.2, 0.0), 6.0, 1.0),
    "brake": Scenario("brake", "带初速度停车", (-0.5, 1.3), 4.0, 0.0),
    "push": Scenario("push", "未知外力干扰", (-1.2, 0.0), 8.0, 1.0, push_start=3.0),
    "switch": Scenario("switch", "目标切换", (-1.2, 0.0), 9.0, 1.0, switch_time=4.0),
    "demo": Scenario("demo", "外力干扰与目标切换", (-1.2, 0.0), 10.0, 1.0,
                     switch_time=6.0, push_start=3.0),
}
