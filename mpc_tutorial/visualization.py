"""MuJoCo 原生窗口与离屏视频共用同一组预测标记。"""
import queue
import time
from pathlib import Path
import mujoco
import numpy as np


def add_prediction(scene, states):
    """橙色小球代表这一次优化预测的未来位置，不是另一个仿真世界。"""
    for i, x in enumerate(states[1:]):
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.025, 0.025, 0.025]),
            pos=np.array([x[0], -0.22, 0.15+0.12*i/max(1, len(states)-1)]),
            mat=np.eye(3).ravel(), rgba=np.array([1.0, 0.56, 0.18, 0.75]),
        )
        scene.ngeom += 1


class Video:
    def __init__(self, path, cfg):
        self.path, self.cfg = Path(path), cfg
        self.renderer = self.writer = None

    def frame(self, frame, plant):
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        if self.renderer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.renderer = mujoco.Renderer(plant.model, height=540, width=960)
            self.writer = imageio.get_writer(str(self.path), fps=1/self.cfg.dt,
                                            codec="libx264", macro_block_size=1)
        self.renderer.update_scene(plant.data, camera="overview")
        add_prediction(self.renderer.scene, frame["plan"].states)
        pixels = self.renderer.render()
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            font = ImageFont.load_default()
        draw.rectangle((0, 0, 960, 85), fill=(17, 27, 38))
        p, v = frame["state"]
        draw.text((18, 10), f"MuJoCo MPC | t={frame['time']:.2f} s | p={p:.3f} m | v={v:.3f} m/s", font=font, fill="white")
        draw.text((18, 34), f"Apply only u[0]={frame['action']:+.2f} N | goal={frame['goal']:+.2f} m | external={frame['force']:+.1f} N", font=font, fill="white")
        draw.text((18, 58), "Blue: actual cart   Green: target   Orange: predicted future positions", font=font, fill=(247, 189, 111))
        self.writer.append_data(np.asarray(image))
        return True

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.renderer is not None:
            self.renderer.close()


class LiveViewer:
    """先创建窗口，再由 simulation 每拍调用 frame；不另建控制线程。"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.viewer = None
        self.keys = queue.SimpleQueue()
        self.paused = False
        self.push_until = -1.0
        self.pending_push = False
        self.deadline = None

    def _poll(self, t):
        while not self.keys.empty():
            key = self.keys.get()
            if key == ord("P"):
                self.pending_push = True
            elif key == ord(" "):
                self.paused = not self.paused

    def force(self, t):
        self._poll(t)
        if self.pending_push:
            self.push_until = t+0.2
            self.pending_push = False
        return 4.0 if t < self.push_until-1e-9 else 0.0

    def frame(self, frame, plant):
        import mujoco.viewer
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(plant.model, plant.data,
                                                       key_callback=self.keys.put)
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = 0
            self.deadline = time.perf_counter()
        if not self.viewer.is_running():
            return False
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            add_prediction(self.viewer.user_scn, frame["plan"].states)
        self.viewer.sync()
        while self.paused and self.viewer.is_running():
            self.viewer.sync()
            self._poll(frame["time"])
            time.sleep(0.02)
            self.deadline = time.perf_counter()
        if not self.viewer.is_running():
            return False
        self.deadline += self.cfg.dt
        remaining = self.deadline-time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        else:
            # 求解/渲染超时不跳过物理步骤，只允许播放慢于实时时间。
            self.deadline = time.perf_counter()
        return True

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
