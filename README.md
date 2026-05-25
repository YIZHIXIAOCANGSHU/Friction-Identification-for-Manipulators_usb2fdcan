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

- 接口：`can0`
- 仲裁波特率：`1000000`
- 数据波特率：`5000000`
- CAN FD：开启

Linux 的 SocketCAN、`ip` 命令和 CAN 内核支持属于主机硬件/系统能力，项目脚本不会安装或修改主机软件环境。

如果不勾选“启动前自动配置 can0”，请先手动配置：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
```

## 按钮

- `全部使能` / `全部失能`：同一个切换按钮，只作用于勾选的电机。使能时会先清错、切 MIT、再使能。
- `一键发送`：只作用于勾选的电机。每个电机会先清错、切 MIT、使能，然后发送当前行的 MIT 五元组。
- `统一发送`：只作用于勾选的电机。所有电机共用“统一 MIT 指令”这一行的五元组。
- `反馈 Rerun`：弹出反馈监视窗口并启动 Rerun viewer，同步读取 `0x11` 到 `0x17` 的反馈帧，显示每个电机的 `position`、`velocity`、`torque`、状态和温度。

## 界面记忆

工具会记住上次输入的 CAN 参数、电机勾选、每行 MIT 参数、统一 MIT 参数和窗口位置。恢复参数只会填入界面，不会自动发送任何硬件命令。

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
