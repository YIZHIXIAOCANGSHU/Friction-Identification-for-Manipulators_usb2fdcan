# MIT 电机一键发送工具

这是一个标准 Python 包结构的 PySide6 桌面工具，用来给达妙电机发送 MIT 模式的五个数字，并可同步读取反馈帧到 Rerun：

- `position`
- `velocity`
- `kp`
- `kd`
- `torque_ff`

## venv 隔离环境

所有 Python 依赖都安装在项目内 `.venv`，不会安装到主机 Python 环境：

```bash
./scripts/setup_venv.sh
```

脚本会执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

运行和测试也都使用 `.venv` 内的解释器。
脚本会清理 `PYTHONPATH` 并设置 `PYTHONNOUSERSITE=1`，避免主机 ROS/系统 Python 包进入项目运行环境。

## 启动

```bash
./run.sh
```

或直接使用虚拟环境入口：

```bash
.venv/bin/mit-sender
```

顶层 `mit_sender_gui.py` 只作为旧入口兼容，正式入口是 `mit-sender`。

## CAN 设置

默认参数：

- 接口：`can0` 或 `can1`，默认 `can0`
- 仲裁波特率：`1000000`
- 数据波特率：`5000000`
- CAN FD：开启

Linux 的 SocketCAN、`ip` 命令和 CAN 内核支持属于主机硬件/系统能力，项目脚本不会安装或修改主机软件环境。

如果不勾选“启动前自动配置接口”，请先手动配置所选接口，例如 `can0`：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
```

也可以在界面中点击 `启动 CAN`。工具会先检测当前选择的接口是否已经存在并处于 `up` 状态；如果已经启动，会直接显示 `CAN 已就绪`，不会再次执行配置命令。只有接口存在但未启动时，才会尝试按当前选择的接口、仲裁波特率和数据波特率执行 SocketCAN 配置命令。
如果接口不存在或未启动，错误弹窗和日志会带出这三行命令，便于复制到终端手动执行。

## 按钮

- `全选` / `全不选`：批量勾选或取消所有目标电机，并保存到下次启动。
- `启动 CAN`：立即配置并拉起所选 `can0`/`can1` 接口。
- `开启读取数据`：按当前页面读取反馈数据，并同步打开 Rerun viewer。批量页读取已勾选电机；单电机调试页会对默认 `0x01..0x07`、当前 CAN ID 和目标 CAN ID 挨个发送使能和全 0 MIT 帧，扫描默认 `0x11..0x17`、当前反馈 ID 和目标反馈 ID 并自动识别当前在线电机。
- `全部使能` / `全部失能`：同一个切换按钮，只作用于勾选的电机。使能时会先清错、切 MIT、再使能。
- `一键发送`：只作用于勾选的电机。每个电机会先清错、切 MIT、使能，然后发送当前行的 MIT 五元组。
- `统一发送`：只作用于勾选的电机。所有电机共用“统一 MIT 指令”这一行的五元组。

## 单电机调试

第二个 Tab 是“单电机调试”，用于总线上只连接一个待配置电机的场景。多电机同时在线时不建议修改 ID。

- `当前 CAN ID`：当前待配置电机的 ID，支持十六进制或十进制。
- `目标 CAN ID`：要写入的新 CAN ID。
- `目标反馈 ID`：要写入的 Master ID/反馈 ID，默认建议值为 `目标 CAN ID + 0x10`，可手动覆盖。
- `开启读取数据`：位于“实时反馈”区域，对默认 CAN ID `0x01..0x07`、当前 CAN ID 和目标 CAN ID 挨个发送使能和全 0 MIT 帧，扫描默认反馈 ID `0x11..0x17`、当前反馈 ID 和目标反馈 ID；读到第一帧后锁定当前电机，之后持续向锁定电机发送全 0 MIT 帧刷新反馈，显示当前 CAN ID、当前反馈 ID 和实时位置/速度/力矩，并直接打开 Rerun viewer，不弹出单独反馈小窗。
- `保存当前位置为零点`：发送零点控制帧。
- `写入 MIT 模式`：写控制模式参数为 MIT，并保存到驱动器，断电保持。
- `设置 ID`：先写 Master ID，再写 CAN ID/ESC_ID，然后保存到驱动器 Flash，断电保持；保存时驱动器可能自动复位。
- `失能后配置单电机`：固定顺序为失能、保存当前位置为零点、写 MIT 模式、写反馈 ID、写 CAN ID、保存参数到 Flash。完成后电机保持失能，需要等待复位完成后手动使能或发送命令。
- `打开说明书`：打开本地 `send/调试助手使用说明书（达妙驱动控制协议）V1.4.pdf`。

执行“保存当前位置为零点”“写入 MIT 模式”“设置 ID”或“一键配置”成功后，工具只记录操作结果，不会自动启动反馈线程或 Rerun。需要查看反馈时，手动点击单电机页“开启读取数据”，它会直接启动 Rerun，不弹出单独反馈小窗。调试页会同步显示当前检测到的 CAN ID/反馈 ID、将写入的目标 CAN ID/反馈 ID、模式，以及实时反馈的 `position`、`velocity`、`torque`。

协议依据：

- 零点控制帧 payload：`FF FF FF FF FF FF FF FE`，命令字 `0xFE`。
- MIT 控制模式：RID `10`，值 `1`。
- Master ID/反馈 ID：RID `7`。
- CAN ID/ESC_ID：RID `8`。
- 写参数帧：`0x7FF [CANID_L, CANID_H, 0x55, RID, DATA0..DATA3]`。
- 保存参数帧：`0x7FF [CANID_L, CANID_H, 0xAA, 0x01]`。

## 界面记忆

工具会记住上次输入的 CAN 参数、电机勾选、每行 MIT 参数、统一 MIT 参数、单电机调试 ID/模式和窗口位置。恢复参数只会填入界面，不会自动发送任何硬件命令。

## 测试

```bash
PYTHONNOUSERSITE=1 env -u PYTHONPATH .venv/bin/python -m unittest discover -v
PYTHONNOUSERSITE=1 env -u PYTHONPATH .venv/bin/python -m compileall src tests
```

## 默认电机

| 电机 | CAN ID | 反馈 ID | 型号 |
| --- | --- | --- | --- |
| 1 | `0x01` | `0x11` | `DM8009` |
| 2 | `0x02` | `0x12` | `DM8009` |
| 3 | `0x03` | `0x13` | `DM4340` |
| 4 | `0x04` | `0x14` | `DM4340` |
| 5 | `0x05` | `0x15` | `DM4310` |
| 6 | `0x06` | `0x16` | `DM4310` |
| 7 | `0x07` | `0x17` | `DM4310` |
