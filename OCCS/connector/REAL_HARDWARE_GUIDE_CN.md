# OCCS 真实硬件接入指南

本文档说明如何在实验室环境中将 OCCS 项目切换为真实硬件（SiliconExtreme 多通道电压源 + Yokogawa AQ6370 OSA）。步骤涵盖依赖安装、配置文件修改、验证脚本的使用以及常见问题排查。

## 1. 前置条件

1. **硬件连接**
   - DAC 通过 USB 转串口连接到实验控制电脑，确认串口号（Windows 通常为 `COMx`，Linux 为 `/dev/ttyUSBx`）。
   - OSA 通过 GPIB、USB 或 LAN 接入，并在 NI MAX 或对应 VISA 工具中确认资源字符串，例如 `GPIB0::1::INSTR`。

2. **软件依赖**
   ```bash
   pip install "pyserial>=3" "pyvisa>=1.13"
   ```
   如使用 NI-VISA，需要确保 NI 驱动程序已经安装并在系统路径中可用。

3. **目标波长网格**
   - 可直接使用优化任务中拟合的波长向量。
   - 或准备一个 CSV 文件（第一列为波长，单位米），供验证脚本读取。

## 2. 配置文件

所有可编辑配置集中在 `OCCS/connector/hardware_config.json`。建议将文件复制一份备份后再修改。关键字段包括：

```json
{
  "real_hardware": {
    "enabled": false,
    "dac": {
      "port": "COM5",
      "channels": [31, 30, 29, 28, 27, 26],
      "baudrate": 115200,
      "timeout": 1.0,
      "write_timeout": 1.0,
      "query_delay": 0.05
    },
    "osa": {
      "resource": "GPIB0::1::INSTR",
      "channel": "a",
      "sensitivity": "high2",
      "speed": "2x"
    },
    "bounds": [0.0, 2.5]
  }
}
```

- 将 `enabled` 设为 `true` 后，前端即可选择真实硬件。
- `dac.channels` 列表长度需要与 DAC 通道数量一致，顺序对应实际接线口编号。
- `bounds` 可以是 `[low, high]` 或 `[[low1, high1], ...]`，用于约束设置电压。
- 如有需要，可新增 `points`、`wavelength_start`、`wavelength_stop` 等字段供验证脚本使用。

> 非编程人员只需编辑此 JSON 文件即可完成配置，无需了解环境变量或命令行参数。

## 3. 使用验证脚本观察仪器响应

项目新增了 `OCCS/connector/hardware_validation.py` 脚本，可用于快速检查硬件链路是否正常。脚本默认读取 `hardware_config.json`，也可以通过 `--config` 指向其他副本。

### 3.1 基本命令

```bash
python -m OCCS.connector.hardware_validation \
  --config OCCS/connector/hardware_config.json \
  --initial-voltage 0,0,0,0,0,0 \
  --test-voltage 1.0,1.0,1.0,1.0,1.0,1.0
```

脚本流程：

1. 按给定电压向量控制 SiliconExtreme，并等待设定的稳定时间（默认 0.2 秒）。
2. 读取回报电压，确认与设置值一致。
3. 触发 OSA 单次扫描，打印输出的功率最小值、最大值及平均值。
4. 如提供 `--test-voltage`，会重复上述流程，便于观察电压改变对光谱的影响。

### 3.2 高级选项

- `--dac-port` / `--osa-resource`: 临时覆盖配置文件中的串口与 VISA 资源信息。
- `--channels`: 指定物理通道编号顺序，便于与硬件实际接线匹配。
- `--wavelength-csv`: 从外部 CSV 载入波长数组，确保与后续优化流程一致。
- `--settle`: 自定义每次施加电压后的等待时间，单位秒。

若观察到 OSA 返回曲线为空，请检查灵敏度、扫描速度以及波长范围设置是否与 MATLAB 脚本一致。

## 4. 将 OCCS 服务切换到真实硬件

1. 编辑 `hardware_config.json`，确保 `enabled: true`，并填入正确的端口、通道及 OSA 资源。
2. 启动 FastAPI 服务或 CLI（不再需要设置额外环境变量）：
   ```bash
   occs-web --host 127.0.0.1 --port 8000
   ```
3. 在 Web UI 中选择 `real` 后端，即可使用真实仪器。

## 5. 常见问题

- **串口占用**：若提示端口已被占用，确认其他程序（MATLAB、串口调试器）已关闭。
- **VISA 连接失败**：确保安装了兼容的 VISA 实现（NI-VISA 或 Keysight IO Suite），并能在官方工具中识别到 OSA。
- **波长插值差异**：RealHardware 会在 OSA 返回的波长网格与请求不完全一致时自动进行线性插值，可通过 `hardware_validation.py` 验证插值效果。
- **真实优化前建议**：先运行验证脚本确认电压设置与响应正常，再执行 OCCS 优化任务，避免在闭环优化过程中排查硬件故障。

如需进一步扩展，例如增加日志记录、自动化扫频或安全限幅，可在 `RealHardware` 中注入自定义控制器或拦截器，保持核心优化代码不变。
