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
## 贝叶斯优化模块概览

除了上述的差分进化示例，本仓库还提供了 `bayes_optimization/` 目录中的一套光芯片电极电压闭环寻优流程。该模块使用仿真器作为数据源，由“物理预标定”“贝叶斯优化”以及“SPSA微调”三个阶段组成，能够在较少测量次数下逼近理想曲线。

执行下列命令即可在命令行运行：

```bash
python OCCS/scripts/run_optimization.py --mode mock
```

该命令会输出最佳电压并在 `data/reports/` 生成波形对比图。

也可以启动前端界面，以交互形式查看优化过程：

```bash
python OCCS/ui/server.py
```

然后打开浏览器访问 `http://localhost:8002` 即可查看结果。
