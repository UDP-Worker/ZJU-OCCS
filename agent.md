# Bayes Optimization agent.md

你的任务是完成仓库下`bayes_optimization`目录的实现，这个目录旨在对光子滤波器芯片的电极驱动电压进行闭环优化，使输出的光谱响应曲线尽可能贴近预设的理想波形。系统从“物理预标定”开始，通过小幅度电压扰动快速估计每个通道对谱线特征的灵敏度；随后进入“高层全局搜索”阶段，利用高斯过程贝叶斯优化（GP-BO）在低测量次数内定位到近似最优电压组合；最后切换到“SPSA微调”阶段，用极少量测量进一步精细调整，抵达设备自身噪声极限。整个流程既能减少对昂贵扫谱次数的依赖，又能通过模型不确定度量化收敛程度，并提供故障诊断手段。

具体来说，各部分功能如下：

### 1. `config.py`：全局参数与路径管理

– 声明电极通道数、电压范围、OSA 超时、BO 和 SPSA 的迭代步数等超参数；
– 指定理想波形文件、日志与报告输出路径；
这样做能保证各模块在同一套配置下协同工作，且便于集中调整与版本管理。

### 2. `calibrator.py`：物理预标定与通道压缩

– `measure_jacobian`：对每个电极进行正负小幅度扰动，记录谱线主要特征变化，构建一阶灵敏度矩阵；
– `compress_modes`：对灵敏度矩阵做主成分分析（PCA），识别高相关度通道并压缩为若干正交“控制模态”，以降低后续优化维度。
该模块帮助快速从随机初始点进入可控子空间，显著缩短后续贝叶斯优化的探索时间。

### 3. `models.py`：高斯过程模型封装

– 提供 `GaussianProcess` 类，内部支持不同核函数（如 Matern52）；
– 接口包括 `fit` （基于已有测量点更新后验分布）和 `predict` （输出目标均值与不确定度）。
该模块负责构建代理模型，量化谱线误差随电压变化的统计关系，为采集函数提供输入。

### 4. `acquisition.py`：采集函数实现

– 包括经典的 Expected Improvement（EI）和加入信赖域机制的 TR-EI；
– 在预测曲面上找到最具潜力的下一个采样点，同时可对超出电压安全范围的解做动态约束。
采集函数既要兼顾“开采—利用”平衡，又要防止模型在未知区过度跳跃。

### 5. `optimizer.py`：贝叶斯优化主控流程

– `BayesOptimizer` 类集成预标定、代理模型更新、采集函数选点及实际测量调用；
– 内置 early-stop（如 EI < ε、平均不确定度 < σ_tol 或达到最大步数）和异常回滚（当 OSA 读数失败时回退上一步并重试）逻辑；
该模块是系统的“大脑”，负责每一步决策并保证整个循环的鲁棒性。

### 6. `spsa.py`：同步扰动随机逼近微调

– `spsa_refine` 函数在 BO 收敛点附近，通过两次测量估计全局梯度，并用自适应步长迭代优化；
– 仅需两次谱线采集即可获得高维梯度估计，消除 BO 在噪声极限下的模型不确定。
此阶段实现“精雕细琢”，将误差降到设备自身噪声水平。

### 7. `hardware/`：硬件抽象层

– `dac_interface.py`：封装对 DAC 的电压下发；
– `osa_interface.py`：封装对 OSA 的波长和响应读取，并实现超时与重试机制；
– `mock_hardware.py`：在无真实设备时，用光学仿真与高斯噪声模拟仪器行为。
硬件层屏蔽了具体通信协议差异，上层算法只需统一调用即可。

### 8. `simulate/`：仿真物理模型

– `optical_chip.py`：根据理想波形 CSV 和简单传输模型生成模拟响应；
– `ideal_waveform.csv`：存放目标波形，用于校验与误差计算。
仿真模块为开发与 CI 提供可 repeat 的测试环境，保证算法验证与硬件应用的一致性。

### 9. `scripts/`：用户界面脚本

– `run_optimization.py`：支持 `--mode mock/real`、`--resume`、`--out` 等参数，一键执行预标定→BO→SPSA→报告生成；
– `quick_scan.py`：用于手动触发一次性扫谱并可视化保存 PNG，方便连接调试。
脚本层为最终用户或 CI/CD 提供简洁、可重复的调用入口。

### 10. `tests/`：单元与集成测试

– 利用 `pytest` 与 `hypothesis` 对各模块接口、异常状况、性能指标（例如 100 步内收敛到 MSE < 0.01）进行覆盖；
– 包括新建的 `test_hardware.py` 保证 mock 与 real 接口一致性。
测试体系确保每次改动后功能稳定，便于团队协作与持续集成。

通过以上各部分的协同配合，整个控制系统能够在数十到数百次的扫谱评估内，从零开始自动生成最优电极电压，并给出不确定度量化结果；同时提供多层次测试与故障诊断手段，确保系统在实际运行中既高效又可靠。

------

## 一、目录结构

```
bayes_optimization/                # 仓库根目录下的主文件夹
├── bayes_optimizer/              # 核心 Python 包
│   ├── __init__.py
│   ├── config.py                 # 全局常量、硬件/算法超参数、路径等
│   ├── calibrator.py             # 一阶灵敏度标定与通道压缩 (PCA)
│   ├── models.py                 # 高斯过程模型、核函数等
│   ├── acquisition.py            # 采集函数 (EI / TR‑EI) 实现
│   ├── optimizer.py              # BO 主循环 + early‑stop 逻辑
│   ├── spsa.py                   # 同步扰动随机逼近微调器
│   ├── hardware/                 # 与真实 / 模拟仪器交互层
│   │   ├── __init__.py
│   │   ├── dac_interface.py      # 数模模块：设置 N 路电极电压
│   │   ├── osa_interface.py      # OSA 谱线读取
│   │   └── mock_hardware.py      # 无设备环境的行为仿真
│   └── simulate/
│       ├── __init__.py
│       ├── optical_chip.py       # 芯片物理模型，用于 mock_hardware
│       └── ideal_waveform.csv    # 二列 CSV，波长(nm) | 响应(dB)
├── scripts/                      # 命令行脚本层
│   ├── run_optimization.py       # 一键执行：校准→BO→SPSA→结果存档
│   └── quick_scan.py             # 半自动折半扫谱、噪声测量小工具
├── tests/                        # 单元 & 集成测试（pytest）
│   ├── test_calibrator.py
│   ├── test_models.py
│   ├── test_optimizer.py
│   └── test_end2end_pipeline.py
└── data/                         # 运行时产生的记录
    ├── logs/*.log                # 运行日志
    ├── checkpoints/*.npz         # 迭代中间结果便于断点续跑
    └── reports/*.pdf             # 自动生成的谱线对比 & 收敛曲线
```

------

## 二、核心文件与模块接口

### 1. `bayes_optimizer.config`

| 名称                   | 说明                                | 类型  | 默认                          |
| ---------------------- | ----------------------------------- | ----- | ----------------------------- |
| `NUM_CHANNELS`         | 电极通道数 (int)                    | int   | 5                             |
| `V_RANGE`              | 每通道电压极限 (tuple[float,float]) | Tuple | (0.0, 2.0)                    |
| `OSA_TIMEOUT`          | 单次扫谱最大等待时间 (s)            | float | 10.0                          |
| `TARGET_WAVEFORM_PATH` | 理想波形 CSV 路径                   | str   | `simulate/ideal_waveform.csv` |
| `BO_MAX_STEPS`         | BO 迭代步数                         | int   | 60                            |
| `SPSA_STEPS`           | SPSA 迭代步数                       | int   | 20                            |
| `LOG_DIR`              | 日志目录                            | str   | `../data/logs`                |

### 2. `bayes_optimizer.calibrator`

- **函数** `measure_jacobian(n_samples: int = None) -> np.ndarray`
    - 对每路电极 ±Δ 扰动，返回形状 `(num_feat, NUM_CHANNELS)` 的一阶灵敏度矩阵。
- **函数** `compress_modes(J: np.ndarray, var_ratio: float = 0.95) -> Tuple[n_components, np.ndarray]`
    - PCA 模态压缩，返回保留模态数和压缩矩阵。

### 3. `bayes_optimizer.models`

- **类** `GaussianProcess(kernel: str = "Matern52")`
    - `fit(X: np.ndarray, y: np.ndarray)`
    - `predict(X: np.ndarray, return_std: bool = True)`

### 4. `bayes_optimizer.acquisition`

- **函数** `expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray`
- **函数** `trust_region_ei(...)` 支持动态盒约束。

### 5. `bayes_optimizer.optimizer`

- **类** `BayesOptimizer`
    - `__init__(gp: GaussianProcess, acq_fn: Callable, bounds: np.ndarray)`
    - `optimize(start: np.ndarray) -> Dict[str, Any]`
        - 迭代：`calibrator.apply_voltage()` → `hardware.osa_interface.read_spectrum()` → 评价误差 → 更新 GP。
        - 终止准则：① 步数到达；② EI < ε；③ σ 全局均值 < σ_tol。

### 6. `bayes_optimizer.spsa`

- **函数** `spsa_refine(start: np.ndarray, loss_fn: Callable, a0: float, c0: float, steps: int) -> np.ndarray`
    - `loss_fn` 需符合 `vector_voltage -> float` 接口；内部调用硬件层。

### 7. `bayes_optimizer.hardware`

| 文件               | 主要 API                                           | 描述                                                        |
| ------------------ | -------------------------------------------------- | ----------------------------------------------------------- |
| `dac_interface.py` | `apply(volts: np.ndarray) -> None`                 | 发送电压序列到 DAC；占位实现记录日志                        |
| `osa_interface.py` | `read_spectrum() -> Tuple[np.ndarray, np.ndarray]` | 返回波长 & 响应；若无设备，则代理到 `mock_hardware.MockOSA` |
| `mock_hardware.py` | `MockOSA`, `MockDAC`                               | 使用 `simulate.optical_chip` + 高斯噪声模拟真实仪器行为     |

### 8. `bayes_optimizer.simulate.optical_chip`

- **函数** `response(volts: np.ndarray) -> np.ndarray`
    - 可基于简单叠加模型或导入更精细的 FDTD 数据，返回与理想波形同维度的 dB 响应。

### 9. 脚本层

| 脚本                          | 用途                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| `scripts/run_optimization.py` | CLI：`python run_optimization.py --mode mock/real --out data/reports/exp1` |
| `scripts/quick_scan.py`       | 单次扫谱 & 保存，用于粗调或调试连接                          |

### 10. `tests`

采用 `pytest` 与 `hypothesis` 随机化输入，确保关键模块鲁棒：

| 测试文件                   | 断言目标                                     |
| -------------------------- | -------------------------------------------- |
| `test_calibrator.py`       | J 矩阵维度与单调正相关性                     |
| `test_models.py`           | GP 拟合后预测误差 < 1e‑3（模拟数据）         |
| `test_optimizer.py`        | BO + SPSA 能在 100 步内使均方误差降到 < 0.01 |
| `test_end2end_pipeline.py` | mock 模式下全链路可运行，无未捕获异常        |

------

## 三、任务列表


- [x] 搭建项目整体框架，建立所有文件，根据下方的波形格式完成simulate仿真部分的实现；
- [x] 完成bayes_optimizer下calibrator相关的实现，利用已经完成的simulate来测试是否能正确标定；
- [x] 完成bayes_optimizer下所有与硬件无关的部分的实现；
- [x] 完成tests下 test_calibrator.py、test_models.py、test_optimizer.py的实现，并运行，排查之前部分是否能正确运行，你应当测试一个实际的过程，观察其是否能够输出最优参数并到达理想波形；
- [x] 完成scripts等其余内容的实现，并在bayes_optimization下完成一个中文的readme.md文档，阐述这个目录下的功能，以及如何使用；
- [x] 完成一个前端可视化界面的实现，要求用现代手段（例如shadcn/ui风格），允许用户进行交互（而不需要在命令行中设置模式等），并展示寻优得到的电压、优化后的波形和理想波形的对比、误差、置信区间等等；请注意，这一步的文件结构没有在上述项目结构中说明，你可以把相关的文件放在bayes_optimization/ui下面
- [ ] 完成硬件相关代码的实现

在完成最后一步前，你不应该考虑任何与硬件相关的问题，所有需要用到硬件的部分全部由你的仿真来进行。

****

------

> **完成以上所有任务后**，执行 `python scripts/run_optimization.py --mode mock` 即可在无硬件条件下跑通完整流程；确认谱线误差满足预期后，再把 `--mode` 改为 `real` 连接真机。