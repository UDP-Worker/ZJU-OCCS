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

### 标定流程

在正式优化前，可通过 `bayes_optimizer.calibrator` 模块对各电极通道进行灵敏度标定：
1. `measure_jacobian()` 对每路电压做微小扰动，获得谱线变化矩阵。
2. `compress_modes()` 根据主成分占比压缩控制维度，减少后续搜索难度。

这些步骤在仿真环境下同样适用，便于模拟真实设备的预标定环节。

### 启动可视化界面

若希望在浏览器中观察优化曲线和最终电压，可执行：
```bash
python ui/server.py
```
然后访问 `http://localhost:8002`。界面中点击“开始优化”后，会自动运行整套流程并绘制优化结果与理想波形的对比。
