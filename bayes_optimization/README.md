# 贝叶斯优化示例

该目录实现了一个针对光学滤波器的参数寻优流程，利用仿真模型在没有真实硬件的情况下演示贝叶斯优化与 SPSA 微调。

## 目录说明
- **bayes_optimizer/**：优化核心模块，包括高斯过程、采集函数以及 SPSA 等实现。
- **scripts/**：命令行脚本，提供优化入口及快速扫谱功能。
- **data/**：运行时生成的日志、检查点和报告文件。
- **tests/**：单元测试，保证各模块在仿真环境下正确工作。

## 使用方法
1. 安装依赖：`pip install numpy scipy scikit-learn matplotlib`
2. 运行优化流程：
   ```bash
   python scripts/run_optimization.py --mode mock
   ```
   脚本会输出最优电压组合与最终误差，并在 `data/reports/` 下生成波形对比图。
3. 仅查看某组电压对应的波形可使用：
   ```bash
   python scripts/quick_scan.py --volts 0.5 0.5 0.5 0.5 0.5 --out scan.png
   ```

所有过程均基于模拟模型，可在无硬件环境下重复运行。
