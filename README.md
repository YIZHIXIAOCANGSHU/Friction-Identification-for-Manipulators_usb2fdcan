# 阶跃力矩扫描

当前主流程已经收缩为单一模式：按电机顺序逐个施加阶跃力矩，并实时读取反馈。

运行行为：

- 从 `0.0 Nm` 开始。
- 每 `1.0 s` 增加一个阶跃，步进为 `0.1 Nm`。
- 只对当前目标电机下发力矩，其余电机保持 `0`。
- 当前目标电机速度绝对值超过 `10 rad/s` 时，立即停止该电机并切换到下一个电机。
- 如果某个电机一直没有超速，则会在达到该电机 `control.max_torque` 后结束该轮。

输出：

- 运行期间会在终端打印当前阶跃和反馈数据。
- 每个电机结束时，会额外打印是否触发了速度极限；如果触发，会输出当时的命令力矩、反馈力矩和速度。
- 每个电机的采样会保存到 `results/runs/<timestamp>_step_torque/.../capture.npz`。

常用命令：

```bash
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode step
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode step --motors 1,3,4
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode step --output results/debug
./run.sh
```
