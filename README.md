# 光芯片仿真与参数优化

本仓库提供一个简单的光学滤波器仿真与差分进化优化示例，
原始代码来自 SRTP 项目中的 Notebook，现已重构为模块化的 Python 代码。

## 目录结构

- `opt_sim/`  核心库，实现器件模型、仿真和优化算法
- `examples/` 示例脚本，展示如何使用差分进化算法优化参数

## 快速开始

```bash
python examples/run_evolution.py
```

脚本会输出找到的最优耦合系数参数。

