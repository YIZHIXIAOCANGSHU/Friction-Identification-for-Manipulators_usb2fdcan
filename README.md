# MIT 电机一键发送工具

这个仓库现在只保留一个最小 PyQt5 界面，用来给达妙电机发送 MIT 模式的五个数字，并可同步读取反馈帧到 Rerun：

- `position`
- `velocity`
- `kp`
- `kd`
- `torque_ff`

## 安装 Qt 和 CAN 工具

Ubuntu 推荐使用系统包：

```bash
sudo apt update
sudo apt install -y python3-pyqt5 python3-pyqt5.qtserialport can-utils
```

如果使用虚拟环境，也可以安装：

```bash
python3 -m pip install -r requirements.txt
```

## 启动

```bash
./run.sh
```

或：

```bash
python3 mit_sender_gui.py
```

## CAN 设置

默认参数：

- 接口：`can0`
- 仲裁波特率：`1000000`
- 数据波特率：`5000000`
- CAN FD：开启

如果不勾选“启动前自动配置 can0”，请先手动配置：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on
sudo ip link set can0 up
```

## 按钮

- `全部使能` / `全部失能`：同一个切换按钮，只作用于勾选的电机。使能时会先清错、切 MIT、再使能。
- `一键发送`：只作用于勾选的电机。每个电机会先清错、切 MIT、使能，然后发送当前行的 MIT 五元组。
- `一键统一发送`：只作用于勾选的电机。所有电机共用“统一 MIT 指令”这一行的五元组。
- `反馈 Rerun`：弹出反馈监视窗口并启动 Rerun viewer，同步读取 `0x11` 到 `0x17` 的反馈帧，显示每个电机的 `position`、`velocity`、`torque`、状态和温度。

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
