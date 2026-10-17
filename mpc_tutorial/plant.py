"""【3 执行】唯一真实推进 MuJoCo 的地方。预测器不能修改这个 data。"""
from pathlib import Path
import mujoco
import numpy as np
from .config import Config


class CartPlant:
    def __init__(self, cfg: Config, x0=(-1.2, 0.0)):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "models/cart.xml"))
        self.model.opt.timestep = cfg.physics_dt
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cart")
        scale = cfg.actual_mass/self.model.body_mass[body]
        self.model.body_mass[body] *= scale
        self.model.body_inertia[body] *= scale
        self.model.actuator_ctrlrange[0] = [-cfg.force_limit, cfg.force_limit]
        self.data = mujoco.MjData(self.model)
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[0], self.data.qvel[0] = x0
        mujoco.mj_forward(self.model, self.data)

    def state(self):
        return np.array([self.data.qpos[0], self.data.qvel[0]])

    def set_goal(self, position):
        self.data.mocap_pos[0] = [position, -0.35, 0.1]
        mujoco.mj_forward(self.model, self.data)

    def step(self, u, external_force=0.0):
        # u 是电机控制；external_force 是未告诉预测器的外界扰动。
        self.data.ctrl[0] = np.clip(u, -self.cfg.force_limit, self.cfg.force_limit)
        self.data.qfrc_applied[0] = external_force
        mujoco.mj_step(self.model, self.data, nstep=self.cfg.substeps)
        self.data.qfrc_applied[0] = 0.0
        return self.state()
