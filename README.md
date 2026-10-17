# 用 MuJoCo 小车，逐拍看懂 MPC

这个项目想让你建立一个具体感觉：**MPC 先设计未来一整段动作，但眼前只执行第一项；实际走了一步后，再根据新状态重新设计未来。**

案例是一辆只沿 x 轴移动的小车。你能控制推力，目标是到指定位置停住，同时考虑推力、速度、位置界限。不需要强化学习、训练模型或下载机器人资源。先把控制逻辑看透，再迁移到倒立摆、机械臂和机器人。

## 1. 先运行，再读推导

项目实际位置：`/home/hw/workspaces/mujoco-mpc-tutorial`。

```bash
cd /home/hw/workspaces/mujoco-mpc-tutorial
uv sync --locked --extra dev

# MuJoCo 原生窗口：蓝色小车，绿色目标，橙色预测位置
uv run mpc-demo run --scenario demo --viewer --out outputs/live
```

这台机器的 `.venv` 已安装并验证。`uv sync` 用于重建相同环境；不必修改系统 Python。

窗口默认运行 10 秒仿真：从 −1.2 m 去 +1 m，3 秒时遭遇一段未知外力，6 秒时目标切到 −1 m。`P` 手动加推力，空格暂停/继续，关闭窗口结束。**橙色球是一拍计划的未来位置，不是额外的小车。** 在目标附近停车时，预测点会聚在一起。

没有窗口也能运行，纯计算不需要 GPU 或图形上下文：

```bash
uv run mpc-demo run --scenario demo --out outputs/demo
# 在浏览器中直接打开 outputs/demo/report.html（完全离线，无 CDN）
```

我已生成 `outputs/demo/report.html`、`outputs/demo/demo.mp4` 和各组对照报告。HTML 的时间滑块允许逐拍查看真实状态、未来预测、整段推力、松弛量和求解耗时。PNG 用于快速概览，JSON 用于自己分析。

导出 MuJoCo 视频：

```bash
MUJOCO_GL=egl uv run mpc-demo run --scenario demo \
  --video outputs/demo/demo.mp4 --out outputs/demo
```

EGL 只影响离屏渲染。如果机器没有 EGL，用正常桌面环境运行并去掉 `MUJOCO_GL=egl`。原生窗口请用默认 GLFW 后端，勿设置成 EGL。当前 Linux 机器已验证原生窗口和 EGL 视频。

## 2. 先看这个反直觉问题：还没到目标，为什么向后推？

```bash
uv run mpc-demo inspect
```

这一拍的条件是：

- 当前位置 −0.5 m，目标 0 m：小车还在目标左边。
- 当前速度 +1.3 m/s：它正在向目标快速运动。
- 质量 1 kg，最大推力 2 N，最大减速度约 2 m/s²。
- 控制周期 0.05 s，预测 40 步，也就是未来 2 s。

默认参数下，第一个最优推力约为 **−1.051 N**。下一步预测位置约为 −0.43645 m，速度约为 +1.24743 m/s。后面计划会继续加大刹车，最终减速停住。

只看位置误差，很容易得出“应该向右加力”。但最大刹车的连续时间近似停车距离是 `v²/(2a) = 1.3²/4 ≈ 0.4225 m`，已经接近剩余的 0.5 m。**现在还不到目标，不代表现在还能继续加速。** 这个公式只是帮助理解，优化器实际使用离散模型、速度代价和推力平滑惩罚，不是照着停车距离公式输出动作。

这里也不能把 PD 概括为“只看位置误差”：带速度反馈的 PD 同样能刹车。本项目的重点是 MPC 如何显式比较一整段未来动作、处理限制和已知未来参考。

## 3. 对照代码，只追这条主线

推荐顺序：

1. [`examples/01_minimal_loop.py`](examples/01_minimal_loop.py)：可运行的最小闭环，先忽略日志与画图。
2. [`config.py`](mpc_tutorial/config.py)：状态权重、预测窗口、约束和物理周期。
3. [`plant.py`](mpc_tutorial/plant.py)：怎么读 MuJoCo 状态、施加推力、推进物理世界。
4. [`dynamics.py`](mpc_tutorial/dynamics.py)：一步模型如何叠成多步预测。
5. [`controller.py`](mpc_tutorial/controller.py)：怎样把“想要什么”写成 QP 并求解。
6. [`simulation.py`](mpc_tutorial/simulation.py)：带外力、对照控制器和日志的完整循环。
7. [`docs/逐步推导.md`](docs/逐步推导.md)：从物理公式到 P、q 和约束矩阵。

核心循环就是：

```python
x_now = plant.state()                       # 读实际状态
reference = scenario.reference(t, dt, N)   # 设未来目标
plan = mpc.solve(x_now, reference)          # 求整个未来动作序列
u = plan.controls[0]                        # 只拿第一项
plant.step(u, external_force=unknown_push)  # 世界走一个控制周期
mpc.record_applied(u)                       # 记录实际执行动作
# 下一拍重新读 x_now，重新求解
```

`MPC.solve()` 不修改 MuJoCo 的真实状态。它用数学模型预测多个候选动作的后果。`CartPlant.step()` 才让真实仿真时间前进。

这解释了一个常见误解：**预测 2 秒不等于先等 2 秒，也不等于执行完 2 秒才重新规划。** 本项目每 0.05 秒都重新优化未来 2 秒。上一拍剩下的动作只是下一拍求解的热启动初值，可以全部改变。

## 4. 本例到底在优化什么？

状态 `x=[p,v]`：位置和速度。输入 `u`：牛顿为单位的推力。每个输入保持 0.05 秒，MuJoCo 内部做 10 个 0.005 秒子步。

一步预测是：

```text
x_(i+1) = A x_i + B u_i
A = [[1, 0.05],         B = [[0.001375],
     [0, 1   ]]              [0.05    ]]
```

位置系数不是连续时间精确离散下的 0.00125。本例刻意匹配 MuJoCo Euler 的 10 个半隐式子步；推导与测试都在项目里。

把未来状态与动作堆起来：

```text
U = [u0, u1, ..., u_(N-1)]
X = [p1, v1, p2, v2, ..., pN, vN]
X = F x_now + G U
```

于是给定一整段 `U`，就能算出一整段 `X`。控制器要找的就是让下面总分最小的 `U`：

```text
J = 中途的位置/速度误差
  + 最后一步的位置/速度误差
  + 推力大小的惩罚
  + 相邻推力变化的惩罚
  + 违反状态限制的惩罚
```

位置误差希望车到目标，速度误差希望车停住，输入代价避免乱用力，平滑代价减少猛变。每个候选动作都受到动力学约束，不能凭空让速度瞬间变零。

本例只停车，所以目标状态是 `[目标位置, 0]`。跟踪运动轨迹时，参考速度通常不能一直设成零。

推力是硬约束：`−2 ≤ u ≤ 2 N`。位置与速度是软约束：`|p| ≤ 1.8 m + s_p`，`|v| ≤ 1.3 m/s + s_v`，`s_p,s_v ≥ 0`，并对松弛量施加强惩罚。**位置与速度的松弛是两组不同单位的量。**

为何允许松弛？如果外力突然把速度推到 2 m/s，而最大刹车能力有限，下一拍不可能立即恢复到 1.3 m/s。强行把不可能实现的要求设成硬约束，就可能无解。软约束让控制器明确付出代价并尽快恢复。它不提供绝对安全保障。

## 5. 按这个顺序做五组实验

### 实验 A：提前刹车

```bash
uv run mpc-demo run --scenario brake --viewer --out outputs/brake
uv run mpc-demo inspect --position -0.5 --velocity 1.3 --target 0
```

在 HTML 报告中暂停到第一拍。你会看到：目标还在前面，第一项却向后推。再看未来推力由负变小、速度渐降，理解“动作服务于整段未来表现”。

### 实验 B：为什么每一步都要重算？

```bash
uv run mpc-demo compare --kind controllers --scenario push --out outputs/controllers
```

包含默认 40 步 MPC、全时长窗口 MPC、全时长一次性开环规划以及一组 PD。

**全时长 MPC 与开环在 t=0 使用相同模型、相同窗口和相同参考，第一项相同。** 区别是之后是否重新读状态、重新优化。3 秒时加入 +4 N、持续 0.2 秒的未知外力。预测器不知道这段扰动；MPC 在实际状态改变后才纠正。

在本机这组参数下：MPC 最终位置误差小于 1 mm，开环约 3.922 m；PD 峰值速度约 2.311 m/s，而 MPC 约 1.300 m/s。精确结果在 `outputs/controllers/metrics.json`。这一组 PD 参数仅作示例，不能据此断言 MPC 总比调好的 PD 好。

注意：轨道范围是软件目标，没有实体挡板，也没有接触。开环跑出画出的轨道后仍会继续运动，正是为了把失控漂移暴露出来。

### 实验 C：窗口长短到底改变了什么？

```bash
uv run mpc-demo compare --kind horizon --scenario reach --out outputs/horizon_reach
uv run mpc-demo compare --kind horizon --scenario brake --out outputs/horizon
```

对照 N=4、12、40。在第一拍比较预测终点的速度、最优动作序列、到达速度和耗时。短窗口只看到 0.2 秒，长窗口看到 2 秒。长窗口算得更多，但不保证每项指标都更好；简单系统、终端代价和权重都可能让短窗口表现不错。

### 实验 D：模型不准会怎样？

```bash
uv run mpc-demo compare --kind mass --scenario push --out outputs/mass
# 单独运行：仿真质量 2 kg，但预测器仍相信是 1 kg
uv run mpc-demo run --mass 2 --model-mass 1 --out outputs/wrong_model
```

报告中比较模型错误和匹配模型。同一个推力在重车上加速更慢，所以上一拍的预测会错。反馈能在这个案例中纠正许多误差，但“每步重算”不等于对任意建模误差都稳定；这里没有做鲁棒 MPC 或在线参数辨识。

### 实验 E：MPC 怎么利用已知的未来信息？

```bash
uv run mpc-demo compare --kind preview --scenario switch --out outputs/preview
```

默认控制器到 t=4 s 才知道目标换了。preview 版本事先知道目标时间表，所以在切换之前就可能开始行动。这种提前行动来自**参考输入中确实给了未来信息**，不是控制器猜到了未知未来。两个版本都不知道外力。

## 6. 适合亲手改的练习

- 将 `config.py` 的 `r_force` 调大十倍：观察是否更节省推力、到达更慢。再观察 `r_delta`，区分“推力小”和“推力变化小”。
- 运行 `run --scenario reach --speed-limit 100`：临时解除有意义的速度限制，观察动作与制动阶段如何变化。
- 改 `terminal_scale` 为 0，再对照短窗口：最后一步的速度为什么更容易被忽略？注意本项目 `Qf=terminal_scale*Q`，不额外叠加一次 Q。
- 给 `scenarios.py` 增加负方向外力或另一组初始速度，检查控制器是否对称响应。
- 在 `MPC.solve()` 中打断点，查看 `free`、`q_u`、`upper`、`result.x`，与 `inspect` 保存的 F/G 矩阵对照。
- 进阶：给物理小车增加阻尼，再推导/辨识对应预测模型；不能只改 MuJoCo XML，却仍声称预测精确匹配。

## 7. 工程边界和验证

这是一阶入门的**线性、受约束、全状态反馈 MPC**，二次规划由 OSQP 求解。MuJoCo 负责物理仿真；其内部接触/约束求解器与这里外层的 MPC 优化器是不同层次。

本例没有状态估计器、接触切换、非线性优化、障碍物避让、严格的实时截止时间、终端不变集合或闭环稳定性证明。终端权重是教学调参，不是稳定性证明。长期恒定外力下也不能据此宣称无静差，通常还需要扰动估计或积分增广。QP 失败时明确报错退出；没有为真实硬件设计后备控制。

```bash
OPENBLAS_NUM_THREADS=1 uv run --extra dev pytest -q
uv run python examples/01_minimal_loop.py
```

测试验证预测与 MuJoCo 一致、多步预测堆叠、QP 与独立有限时域 Riccati 解一致、约束外初始状态可以用松弛恢复、求解不推进真实仿真、未知外力下闭环能恢复而开环漂移、模型质量误差可见、提前刹车与无效参数检查。当前 13 项通过。

规划耗时是本机测量值，包含 Python 更新、热启动和 QP 求解，不包含 MuJoCo/渲染；不是实时保证。`OPENBLAS_NUM_THREADS=1` 能避免小矩阵运算被多线程调度开销拖慢，也有利于复现实验耗时。

## 8. 想迁移到机器人时，保留哪些思想？

保持相同的五步闭环：读状态、建立未来参考、预测并优化、执行第一项、重测。改变的是状态/输入定义、动力学模型、代价、约束与优化算法。

倒立摆有非线性，可以围绕直立点线性化做局部 MPC，也可用非线性模型做 NMPC。机械臂会增加关节/扭矩/障碍限制。接触式机器人还涉及接触模式与摩擦约束，难度明显更高。不要把本例的 F/G 矩阵直接套过去，但可以保留整套思路与工程分层。

参考：

- [MuJoCo Python 官方文档](https://mujoco.readthedocs.io/en/stable/python.html)：模型、状态、物理推进与原生 viewer。
- [MuJoCo 计算流程](https://mujoco.readthedocs.io/en/stable/computation/index.html)：状态与数值积分。
- [OSQP 官方 MPC 示例](https://osqp.org/docs/examples/mpc.html)：另一种保留状态变量的 QP 写法；本项目采用消元后的 condensed 写法。
