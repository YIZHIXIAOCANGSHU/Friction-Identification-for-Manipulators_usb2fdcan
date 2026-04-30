# MIT `identify-all` / `compensation`

当前正式入口包含两个模式：

- `identify-all`：起转辨识 -> 稳态摩擦辨识 -> 惯量辨识 -> 验证汇总
- `compensation`：加载最近一次辨识得到的电机模型，实时计算并下发补偿力矩

`identify-all` 实验链路固定为：

- `Phase 0` 预检查
- `Phase 1` 起转 / 静摩擦扫描
- `Phase 2` 低速 / 静摩擦过渡区采集
- `Phase 3` MIT 定速度摩擦辨识
- `Phase 4` MIT 速度斜坡惯量辨识
- `Phase 4b` dynamic MIT 位置/速度轨迹采集（默认接入，速度上限低于辨识预算）
- `Phase 5` `piecewise_static_linear_v1` 生成、预算校验、验证与汇总

关键约束：

- 底层发包全部收口到 `send/`
- 默认电机型号映射为 `DM8009, DM8009, DM4340, DM4340, DM4310, DM4310, DM4310`
- 全流程统一硬停止条件：`abs(velocity) >= 10 rad/s`
- 采集生成使用 `identification.generation_safety_margin_ratio` 得到辨识速度预算，超预算配置直接报错，不做事后缩放或裁剪。
- 起转辨识步进固定为 `0.01 Nm`

配置重点：

- `transport.motor_types` 必须和现场真实电机一致，否则 MIT 缩放会错。
- `safety.hard_speed_abort_abs` 默认 `10.0`
- `breakaway.torque_step` 默认 `0.01`
- `mit_velocity.kd_speed` 支持逐电机配置
- `low_speed.speed_points` 默认 `[0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50, 0.75, 1.00]`，按速度幅值成对分配 train/valid。
- `low_speed.micro_motion_velocity_limit` 默认 `0.2 rad/s`，微动 MIT 状态增益用 `micro_motion_kp=1.5`、`micro_motion_kd=0.25`，不复用 dynamic MIT 增益。
- `inertia.waypoints` 控制惯量阶段速度斜坡 waypoint。
- `identification.steady_speed_points` 默认最高到 `6.0 rad/s`；10 rad/s 仍是硬峰值限制，不作为默认辨识验证点。
- `identification.generation_safety_margin_ratio` 默认 `0.80`，所有 speed-hold、low-speed、inertia 和 dynamic MIT 命令都必须低于该预算。
- `identification.joint_*_weight` 控制 speed-hold / inertia / dynamic MIT 在联合拟合中的权重。
- `identification.min_tracking_ratio` / `identification.max_steady_velocity_std_ratio` 用于筛掉跟踪差、平台不稳的速度保持段。
- `identification.friction_rmse_publish_threshold` / `identification.inertia_rmse_publish_threshold` 控制发布门槛。
- `dynamic_mit.enabled` 默认 `true`；作为 `identify-all` 的联合拟合数据来源之一。
- `dynamic_mit.velocity_limit`、`position_amplitude`、`frequency_hz` / `frequency_range_hz`、`kp`、`kd` 是 dynamic MIT 优先调参项；组合超速会在运行前报错。
- `identification.min_publishable_rounds` 默认 `2`，最新辨识尝试会写入 `latest_motor_parameters.json`；未达发布条件时 `publish_status` 会是 `not_published` 或 `rejected`，旧的已发布模型只保存在 `previous_published_model` 里作参考。
- `compensation.require_published_model` 默认 `true`，补偿模式默认只允许加载已发布模型。
- `compensation.torque_limit_ratio` 默认 `0.25`，`torque_slew_rate_nm_s` 默认 `2.0`，用于降低在线补偿激进度；不要提高 10 rad/s 硬峰值来规避软停止。
- `compensation.static_velocity_threshold_rad_s` / `static_transition_velocity_rad_s` 控制分段静摩擦过渡；发布前会做补偿力矩 envelope 校验，超预算模型会被拒绝发布。
- `output.latest_parameters_json_filename` 默认 `latest_motor_parameters.json`

常用命令：

```bash
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify-all
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode identify-all --motors 1,3,4
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode compensation --motors 3
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode breakaway
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode speed-hold
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode inertia
python3 -m friction_identification_core --config friction_identification_core/default.yaml --mode dynamic-mit --motors 1
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
  - 已发布模型会带 `model_version`、`model_kind`、`fit_method`、`source_phases`、`publish_status`、`publish_detail`、`accepted_round_count`、`selected_rounds`、`confidence`、`quality_flags`
  - 导出的 `friction_model.kind` 固定为 `piecewise_static_linear_v1`，嵌入式参数位于 `export_models.embedded_piecewise_linear_friction`
  - 若本轮辨识未达到发布条件，默认补偿会拒绝使用该未发布模型；确实需要现场强制验证时，可把 `compensation.require_published_model` 改为 `false`

模块职责：

- `workflow.py`：运行调度、结果落盘、CLI runner。
- `capture.py`：采样 buffer、命令/反馈记录、Rerun live sample 桥接。
- `safety.py`：硬速度停止、软停止、等待静止、precheck、zero/disable 保护。
- `phases/breakaway.py`：起转/静摩擦扫描。
- `phases/speed_hold.py`：定速度摩擦辨识采集。
- `phases/inertia.py`：速度斜坡惯量辨识采集。
- `phases/dynamic_mit.py`：MIT 位置/速度轨迹采集。
- `compensation.py`：latest 模型加载与在线补偿控制。

dynamic MIT 采集：

- 支持 `sine`、`chirp`、`trapezoid_velocity` 三类轨迹，生成 `position_cmd`、`velocity_cmd`、`acceleration_cmd`。
- 使用 MIT 模式发送 `position=position_cmd`、`velocity=velocity_cmd`、`kp=dynamic_mit.kp`、`kd=dynamic_mit.kd`、`torque_ff=0`。
- 每条目标电机反馈会记录命令、实际位置/速度/力矩、位置/速度误差、状态标记、`used_for_fit` 和诊断量 `tau_mit_est`。
- 拟合默认使用 `torque_feedback`；只有 `dynamic_mit.use_mit_estimated_torque: true` 时才允许用 `tau_mit_est`。
- `identify-all` 会把 speed-hold 平台点、inertia 段和 dynamic MIT 有效样本用于联合候选拟合，最终在线补偿和导出统一使用 `piecewise_static_linear_v1`。
- Rerun 记录会按 `Overview / Feedback Frames / Motors / Summary` 组织，`Motors` 内每个电机一个子页面。

SocketCAN / 达妙协议：

- 上层 `friction_identification_core` 不再直接拼 MIT 参数。
- `send/damiao.py` 负责 MIT 力矩、MIT 速度、原生速度模式、使能、失能、清错、零命令和反馈解码。
- MIT 速度控制语义固定为 `position=0, kp=0, kd=kd_speed, velocity=v_des, torque_ff=0`
- MIT 力矩控制语义固定为 `position=0, kp=0, kd=0, velocity=0, torque=t_ff`
- MIT 状态控制语义固定为 `send_mit_state(motor_id, position, velocity, kp, kd, torque_ff=0)`

验证：

```bash
python3 -m unittest discover -s tests -q
```
