# OCCS：光芯片曲线寻优工具

[English](./README.en.md) | [简体中文](./README.zh-CN.md)

轻量级的光芯片响应仿真 + 贝叶斯优化工具包，用于调节电压，使测得光谱尽可能匹配目标曲线。项目模块化、含单元测试，并提供优化过程的日志与可视化（包含高斯过程不确定度）。

## 主要特性

- 模拟硬件与可复现的简化光学响应模型
- 曲线相似度目标：小范围对齐、稳健损失（Huber/L2）、可选分区权重
- 基于 skopt 的贝叶斯优化；逐迭代记录 GP 最大不确定度
- 输出图表（损失、running min、GP 不确定度）与 CSV 日志

## 项目结构

- `OCCS/`
  - `connector/`：硬件适配层
    - `mock_hardware.py`：内存模拟设备（测试/示例使用）
    - `real_hardware.py`：真实硬件接口占位（API 说明）
  - `simulate/`：简化的光学响应模型（`get_response`）
  - `optimizer/`：目标函数、优化封装、可视化
    - `objective.py`：曲线/硬件目标与 CSV 构造器
    - `optimizer.py`：`skopt.Optimizer` 封装，记录 GP 不确定度
    - `viz.py`：绘制损失与不确定度，导出 CSV
  - `data/optimization/`：测试/示例的默认输出目录（图与日志）
- `tests/`：基于 Pytest 的测试用例（运行时也会生成示例输出）
- `environment.yml`：Conda 环境文件（依赖包含 NumPy、scikit-optimize、Matplotlib、PyTest 等）

## 快速开始

1）安装环境

```bash
# 使用 conda/mamba
mamba env create -f environment.yml  # 或：conda env create -f environment.yml
mamba activate ZJU-OCCS              # 或：conda activate ZJU-OCCS
```

2）运行测试（同时生成示例输出）

```bash
pytest -q
```

输出位于 `OCCS/data/optimization/`：

- `loss_curve_*.png`：损失随迭代变化 + running min；若可用，会叠加 GP 最大标准差
- `log_*.csv`：每次迭代的电压、损失、delta_nm 以及不确定度（`gp_max_std`, `gp_max_var`）

3）最小示例

```python
import numpy as np
from OCCS.connector import MockHardware
from OCCS.optimizer.objective import create_hardware_objective
from OCCS.optimizer.optimizer import BayesianOptimizer
from OCCS.optimizer.viz import save_loss_history_plot, save_uncertainty_history_plot, save_history_csv

lam = np.linspace(1.55e-6, 1.56e-6, 200)
bounds = [(-1.0, 1.0)] * 3
hw = MockHardware(dac_size=3, wavelength=lam, voltage_bounds=bounds)

# 从两行 CSV 构建目标（第一行：波长；第二行：目标值）
obj = create_hardware_objective(hw, target_csv_path="OCCS/data/ideal_waveform.csv", M=200)

bo = BayesianOptimizer(obj, dimensions=bounds, base_estimator="GP", acq_func="EI", random_state=42)
result = bo.run(n_calls=30, x0=[0.0, 0.0, 0.0])

save_loss_history_plot(result, "OCCS/data/optimization/loss_curve_example.png", title="BO Loss")
save_uncertainty_history_plot(result, "OCCS/data/optimization/gp_uncertainty.png", metric="std", title="GP Max Std")
save_history_csv(result, "OCCS/data/optimization/log_example.csv")
```

## 实现说明

- GP 最大不确定度通过在边界内随机采样近似估计；模型尚未拟合的早期步骤可能为 NaN。
- 目标函数在归一形状上度量差异，并支持小范围波长对齐、Huber 稳健损失与分区加权配置；参见 `ObjectiveConfig`。

## 贡献指南

- 代码风格：NumPy 风格文档字符串；模块小而专注；配套测试。
- 测试：`pytest -q`。新增功能请附加测试和最小文档。

## 许可证

许可证信息见仓库根目录的 LICENSE 文件。

