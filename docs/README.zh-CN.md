# OCCS：光芯片控制与寻优（Web UI + FastAPI）

[English](./README.en.md) | [简体中文](./README.zh-CN.md)

OCCS 面向实验工程师提供一个交互式网页界面与 Python 服务层，可连接模拟或真实硬件，实时展示波形与优化过程，并通过贝叶斯优化自动搜索使光谱逼近目标曲线。核心算法与模块保持可复用，并附有测试用例。

## 主要特性

- Web 前端（Vite + React + TS）：实时波形与 loss 曲线、诊断信息
- FastAPI 后端：REST + WebSocket，支持会话与实时推送
- 手动电压与边界裁剪；CSV 历史下载
- 可复用核心：仿真、目标函数与基于 skopt 的贝叶斯优化

## 架构概览

- `OCCS/optimizer`：目标函数、优化封装、离线可视化
- `OCCS/connector`：硬件接口（`MockHardware` 与 `RealHardware` 占位）
- `OCCS/service`：FastAPI 应用、会话管理、硬件工厂
  - REST：创建会话、写电压、读波形、启动/停止优化、导出历史
  - WebSocket：推送进度、波形快照与状态
- `OCCS/webui`：单页应用（构建后由后端静态托管）

## 快速开始

### 1）安装依赖

使用 Conda/Mamba（推荐）：

```bash
mamba env create -f environment.yml  # 或：conda env create -f environment.yml
mamba activate ZJU-OCCS

# 启动 Web 服务所需（方案 A）
pip install "OCCS[web]"  # 安装 fastapi、uvicorn、python-multipart
# 或（方案 B）
pip install fastapi uvicorn python-multipart
```

或使用 pip（Python >= 3.10）：

```bash
pip install -e .
pip install numpy scipy scikit-learn scikit-optimize matplotlib
pip install "OCCS[web]"  # 或：fastapi uvicorn python-multipart
```

### 2）启动后端

```bash
occs-web --host 127.0.0.1 --port 8000
# 或：python -m OCCS.service.app
```

接口位于 `http://127.0.0.1:8000/api`。

### 3）运行前端（开发或生产）

开发（Vite 代理热更新）：

```bash
cd OCCS/webui
npm install  # 或：pnpm install / yarn
npm run dev
# 打开 http://127.0.0.1:5173 （自动代理到后端 /api 与 WS）
```

生产构建（由后端托管）：

```bash
cd OCCS/webui
npm run build
# 重启后端；访问 http://127.0.0.1:8000/
```

### 4）Docker

本地构建并运行：

```bash
docker build -t occs-web:local .
docker run --rm -p 8000:8000 occs-web:local
# 打开 http://127.0.0.1:8000/
```

启用真实硬件（可选）：

```bash
docker run --rm -e OCCS_REAL_AVAILABLE=1 -p 8000:8000 occs-web:local
```

使用 GitHub Releases 提供的预构建镜像：

```bash
# 1）从 Release 附件中下载 occs-web.tar.gz
# 2）加载镜像
docker load -i occs-web.tar.gz

# docker 会打印镜像名与标签（例如 occs-web:0.1.1）
# 3）运行（请将 <image:tag> 替换为实际值）
docker run --rm -p 8000:8000 <image:tag>

# 可选：启用真实硬件并持久化上传目录
docker run --rm -e OCCS_REAL_AVAILABLE=1 \
  -v $(pwd)/uploads:/tmp/occs_uploads \
  -p 8000:8000 <image:tag>
```

### 5）使用流程

- 选择后端（默认 mock）。真实硬件默认关闭，需将环境变量 `OCCS_REAL_AVAILABLE=1` 以启用。
- 配置波长网格与电压边界；可上传理想目标 CSV。
- 创建会话，手动下发电压并刷新波形。
- 启动优化，实时查看 loss 与诊断；可随时下载历史 CSV。

## API 一览（REST）

- `GET /api/backends` → 可用后端
- `POST /api/session` → 创建会话，返回 `{ session_id }`
- `GET /api/session/{id}/status` → 运行态、迭代数、最优损失、当前最优 x
- `DELETE /api/session/{id}` → 关闭会话
- `POST /api/session/{id}/voltages` → 写电压
- `GET /api/session/{id}/response` → 波形 `{ lambda, signal, target }`
- `POST /api/session/{id}/optimize/start` → 启动优化
- `POST /api/session/{id}/optimize/stop` → 停止优化
- `GET /api/session/{id}/history(.csv)` → 获取/导出历史

WebSocket：`GET /api/session/{id}/stream` → 事件包含 `status`、`progress`、`waveform`、`done`、`error`。

## 开发与测试

- 运行测试（会生成示例图与 CSV）：`pytest -q`
- 代码风格：保持简洁、为新增功能补充测试

## 许可证

许可证信息见仓库根目录的 LICENSE 文件。
