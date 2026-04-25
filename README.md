# MIT `identify-all` / `compensation`

当前正式入口包含两个模式：

- `identify-all`：起转辨识 -> 稳态摩擦辨识 -> 惯量辨识 -> 验证汇总
- `compensation`：加载最近一次辨识得到的电机模型，实时计算并下发补偿力矩

`identify-all` 实验链路固定为：

- `Phase 0` 预检查
- `Phase 1` 起转 / 静摩擦扫描
- `Phase 2` MIT 定速度摩擦辨识
- `Phase 3` MIT 速度斜坡惯量辨识
- `Phase 4` 验证与汇总

关键约束：

- 底层发包全部收口到 `send/`
- 默认电机型号映射为 `DM8009, DM8009, DM4340, DM4340, DM4310, DM4310, DM4310`
- 全流程统一硬停止条件：`abs(velocity) >= 10 rad/s`
- 起转辨识步进固定为 `0.01 Nm`

配置重点：

- `transport.motor_types` 必须和现场真实电机一致，否则 MIT 缩放会错。
- `safety.hard_speed_abort_abs` 默认 `10.0`
- `breakaway.torque_step` 默认 `0.01`
- `mit_velocity.kd_speed` 支持逐电机配置
- `output.latest_parameters_json_filename` 默认 `latest_motor_parameters.json`

常用命令：

```bash
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify-all
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify-all --motors 1,3,4
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode compensation --motors 3
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode breakaway
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode speed-hold
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode inertia
./run.sh
```

输出：

- 每轮原始采样会保存到 `results/runs/<timestamp>_<mode>/group_xx/motor_xx/capture.npz`
- 每轮辨识结果会保存到 `results/runs/<timestamp>_<mode>/group_xx/motor_xx/identification.npz`
- 汇总结果会保存到：
  - `results/runs/<timestamp>_<mode>/summary/hardware_identification_summary.npz`
  - `results/runs/<timestamp>_<mode>/summary/hardware_identification_summary.csv`
  - `results/runs/<timestamp>_<mode>/summary/hardware_identification_summary.md`
- 最新模型登记会保存到 `results/latest_motor_parameters.json`

SocketCAN / 达妙协议：

- 上层 `friction_identification_core` 不再直接拼 MIT 参数。
- `send/damiao.py` 负责 MIT 力矩、MIT 速度、原生速度模式、使能、失能、清错、零命令和反馈解码。
- MIT 速度控制语义固定为 `position=0, kp=0, kd=kd_speed, velocity=v_des, torque_ff=0`
- MIT 力矩控制语义固定为 `position=0, kp=0, kd=0, velocity=0, torque=t_ff`
