 # OCCS Web UI 开发计划（AGENTS）

 本文件用于规划并跟踪为 OCCS 增加一个基于 React 的网页界面的工作。目标是在不重写现有优化器/目标函数/连接器核心模块的前提下，新增一个后端服务层（FastAPI）与一个前端（Vite + React + TypeScript），实现：
 - 选择后端：真实硬件或模拟器
 - 展示波形：显示当前采集到的波形与目标曲线
 - 手动下发电压：通过界面输入并应用到硬件
 - 启动优化：可配置参数，实时展示 loss 曲线与相关诊断（xi/kappa、GP 不确定度）

 ---

 ## 背景与目标
 - 现有能力：`OCCS.optimizer` 提供 BO 优化与可视化；`OCCS.connector` 提供模拟与真实硬件接口（真实为占位）；`tests/test_optimizer.py` 验证优化有效。
 - 新增目标：提供交互式 Web 界面，面向实验工程师，支持模拟/真实两种数据源，提供实时曲线与控制。

 ---

 ## 总体架构
 - 后端（Python/FastAPI）：目录 `OCCS/service`
   - 提供 REST 接口用于会话、硬件操作与历史查询
   - 提供 WebSocket 实时推送迭代进度与波形
   - 会话管理：一个会话绑定一次硬件实例 + 一个优化器；支持启动/停止优化
  - 统一硬件接口：当前 `MockHardware` 与 `RealHardware` 均采用 `get_response()`，服务层可直接依赖统一方法；如需附加保护（额外裁剪/日志/指标），可通过轻量适配器扩展
   - 静态托管前端构建产物
 - 前端（Vite + React + TS）：目录 `OCCS/webui`
   - 单页应用（SPA），使用 ECharts 或 Chart.js 绘制波形与 loss 曲线
   - 通过 REST/WS 与后端通讯
   - 组件化：后端选择/会话表单/电压输入与应用/优化控制/图表/状态栏
 - 复用核心：优化器 `BayesianOptimizer`、目标封装 `create_hardware_objective`、可选 PNG/CSV 导出 `viz.py`

 ---

 ## 目录与文件规划
 - `OCCS/service/app.py`：创建 FastAPI 应用、注册路由、挂载静态文件、维护会话字典
- `OCCS/service/hardware.py`：硬件工厂（可选适配层）
  - `list_backends()`：返回可用后端
  - `make_hardware(backend, ...)`：构造 Mock/Real 实例（两者已统一 `get_response()`）
  - （可选）`HardwareAdapter`：用于边界保护、额外日志或统计（非必须）
 - `OCCS/service/models.py`：Pydantic 模型（请求/响应/事件）
 - `OCCS/service/session.py`：会话与优化运行器
   - `OptimizerSession`：管理硬件、目标、优化器、历史；逐步调用 `optimizer.step()` 并推送事件
 - `OCCS/service/events.py`：事件类型常量与序列化工具
 - `OCCS/cli.py`：命令入口 `occs-web`，启动 `uvicorn`
 - `OCCS/webui/`（React 应用）
   - `index.html`、`vite.config.ts`
   - `src/main.tsx`、`src/App.tsx`
   - `src/api/client.ts`（REST）与 `src/api/ws.ts`（WebSocket hook）
   - `src/components/*`：`BackendSelector`、`SessionForm`、`VoltagePanel`、`WaveformChart`、`LossChart`、`OptimizerControls`、`StatusBar`
   - `src/types.ts`：与后端 `models.py` 对齐的类型

 ---

 ## 接口设计（REST/WS）
 - 会话与状态
   - `GET /api/backends` → `[{name:"mock", available:true}, {name:"real", available:false}]`
   - `POST /api/session` → 创建会话；请求体：
     - `backend: "mock" | "real"`
     - `dac_size: number`
     - `wavelength: number[]` 或 `{ start:number, stop:number, M:number }`
     - `bounds: [low, high][]`（长度 = `dac_size`，或单个 `[low, high]` 广播）
     - `target_csv_path: string`
     - 可选 `optimizer: { acq_func?: string, random_state?: number }`
     - 返回 `{ session_id: string }`
   - `GET /api/session/{id}/status` → `{ running:boolean, iter:number, best_loss:number|null, x?:number[] }`
   - `DELETE /api/session/{id}` → 关闭并清理会话
 - 硬件操作
   - `POST /api/session/{id}/voltages` → `{"volts": number[]}`；返回 `{ ok:true }`
  - `GET /api/session/{id}/response` → `{ lambda:number[], signal:number[], target:number[] }`
 - 优化管理
   - `POST /api/session/{id}/optimize/start` → `{"n_calls": number, "x0"?: number[], "acq_func"?: string, "random_state"?: number }`
   - `POST /api/session/{id}/optimize/stop` → `{ ok:true }`
   - `GET /api/session/{id}/history` → 返回与 `BayesianOptimizer.run` 类似的结果 `{ history:[{x, loss, diag}], best_loss, best_x }`
 - 实时流（WebSocket）
   - `WS /api/session/{id}/stream`
   - 事件：
     - `progress`：`{"type":"progress","iter":3,"loss":0.123,"running_min":0.120,"xi":0.05,"kappa":null,"gp_max_std":0.21,"x":[...]}`
     - `waveform`：`{"type":"waveform","lambda":[...],"signal":[...],"target":[...]}`
     - `status`：`{"type":"status","running":true,"iter":3,"best_loss":0.12}`
     - `done`：`{"type":"done","best_loss":0.1}` / `error`：`{"type":"error","message":"..."}`

 ---

 ## 事件与模型（核心字段）
 - `CreateSessionRequest`：见上 `POST /api/session`
 - `StatusResponse`：`running, iter, best_loss, x`
 - `ProgressEvent`：`iter, loss, running_min, xi?, kappa?, gp_max_std?, gp_max_var?, x`
 - `WaveformPayload`：`lambda, signal, target`

 ---

 ## 前端交互流程
 - 启动页面：
   1. `GET /api/backends` 获取可用后端
   2. 选择后端与参数（`dac_size/wavelength/bounds/target_csv_path`）→ `POST /api/session`
   3. 保存 `session_id`；连接 `WS /api/session/{id}/stream`
 - 手动控制：
   - 在 `VoltagePanel` 输入数组 → `POST /api/session/{id}/voltages`；随后 `GET /api/session/{id}/response` 刷新波形
 - 优化：
   - `POST /api/session/{id}/optimize/start`；WS 持续收到 `progress`，前端更新 loss 与诊断
   - `POST /api/session/{id}/optimize/stop` 可终止
   - `GET /api/session/{id}/history` 可下载 CSV/展示结果

 ---

 ## 风险与决策
 - 接口统一：`MockHardware` 已与 `RealHardware` 一致采用 `get_response()`，无需额外命名适配；如需增强（裁剪/日志），可在服务层增加可选适配器。
 - 逐步优化 vs 一次性 `run()`：为实时更新，采用循环 `optimizer.step()` 并在每步推送事件，必要时可复用 `viz.py` 导出最终 PNG/CSV。
 - 会话并发：每个会话同一时刻仅允许一个优化任务运行；多 WS 订阅者共享广播。
 - 安全限制：电压写入会被硬件/适配器按 bounds 进行裁剪；真实硬件默认标记为 `available:false`，需显式配置启用。

 ---

 ## 里程碑与验收
 - M1 后端雏形：Mock 模式创建会话、手动电压、`GET /response` 返回波形
 - M2 实时优化：WS 推进度，前端实时 loss 曲线
 - M3 参数化与导出：支持 acq_func/random_state；提供 CSV/PNG 下载
 - M4 真实硬件：`RealHardware` 对接与启用开关（不改前端）

 验收标准：
 - 关键接口返回符合约定；Mock 模式端到端可用；优化过程中 loss 明显下降；前端可交互、可视化稳定（>200 点仍清晰）。

 ---

 ## 开发运行说明（建议）
 - 本地开发：
   - 后端：`occs-web --host 127.0.0.1 --port 8000`（新增 CLI 入口）
   - 前端：`cd OCCS/webui && pnpm dev`（开发期走前端代理到后端 `/api`/`/ws`），构建后产物由后端静态托管
 - 构建与托管：
   - `pnpm build` 产物到 `OCCS/webui/dist`，`app.py` 挂载为根路径（`/`）

 ---

 ## 任务清单（进度跟踪）

 ### Phase 0 文档与准备
 - [x] 创建 AGENTS.md（本文件）

### Phase 1 后端框架（已完成）
- [x] 新建 `OCCS/service` 目录与模块骨架
- [x] `hardware.py`：`list_backends()`、`make_hardware()`（统一 `get_response()` 已就绪）
- [x] （可选）`HardwareAdapter`：保留为可选增强（未必需）
- [x] `models.py`：轻量 schema/归一化工具（Phase 1 以 dataclass/函数实现）
- [x] `session.py`：`OptimizerSession`（创建硬件与目标、手动电压、读取波形、基本状态）
- [x] `app.py`：`create_app()` 工厂 + `/api/health`（FastAPI 懒导入）
- [x] `OCCS/cli.py`：`occs-web` 入口（缺少 uvicorn/fastapi 时给出友好提示并返回错误码）
- [x] `pyproject.toml`：添加 console-script `occs-web = OCCS.cli:web_main`
- [x] 单元测试：`tests/test_service_hardware.py`、`tests/test_service_session.py`、`tests/test_cli_web.py`

### Phase 2 REST/WS 接口与健壮性（进行中）
- [ ] REST：`/api/backends`、`/api/session`（POST/GET/DELETE）、`/voltages`、`/response`、`/optimize/start|stop`、`/history`
- [ ] WS：`/api/session/{id}/stream`（progress/waveform/status/done/error）
- [ ] 校验与错误处理（bounds、维度、文件路径）
- [ ] CORS/跨域与简单鉴权（可选）

 ### Phase 3 前端工程
 - [ ] Vite + React + TS 脚手架（`OCCS/webui`）
 - [ ] API 客户端与 WS Hook
 - [ ] 组件：`BackendSelector`、`SessionForm`、`VoltagePanel`、`WaveformChart`、`LossChart`、`OptimizerControls`、`StatusBar`
 - [ ] 主题与基础样式

 ### Phase 4 联调与交付
 - [ ] 手动电压 → 波形展示流打通
 - [ ] 启动优化 → 实时曲线/诊断更新
 - [ ] 历史导出 CSV（可调用 `viz.save_history_csv`）与截图/PNG（可选）
 - [ ] README/Docs 更新与使用示例

 ### Phase 5 硬件接入（可选）
 - [ ] `RealHardware` 接入与适配器封装
 - [ ] 安全限幅与运行开关

 ---

 ## 附注
 - 现有 `HardwareObjective` 调用 `MockHardware.get_simulated_response()`；服务层通过 `HardwareAdapter` 统一实现 `get_response()`，并在 `create_hardware_objective` 前注入适配器，从而无需改动优化/目标模块。
 - `viz.py` 仍用于导出报告型 PNG/CSV；在线实时图表优先前端绘制，避免对后端图形库强依赖。
