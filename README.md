# ⚡ ComfyUI + MiniMax-H3 (Colab G4 96GB) 一行命令极速部署

专为 **Google Colab G4 (RTX PRO 6000 Blackwell 96GB VRAM / ~177GB RAM)** 打造的 ComfyUI + `Comfy-Org/MiniMax-H3` (8-Step Turbo) 一键恢复环境。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nfft894htno32/comfyui-h3-colab/blob/main/ComfyUI_G4_OneClick.ipynb)

## 1. 一行命令快速部署（任意 Colab 代码格粘贴即跑）

```python
import urllib.request; exec(urllib.request.urlopen("https://raw.githubusercontent.com/nfft894htno32/comfyui-h3-colab/main/run.py").read().decode(), globals())
```

或者直接收藏一键启动书签（预设 G4 GPU，单单元格一键运行）：  
👉 https://colab.research.google.com/github/nfft894htno32/comfyui-h3-colab/blob/main/ComfyUI_G4_OneClick.ipynb

---

## 2. 核心加速机制

1. **网盘单包离线快照 (`env_cache/*.tar.zst`)**：
   - **首次运行**：5 线程并发克隆 ComfyUI 与全部核心节点 + `uv` 极速安装 Blackwell 加速库（`comfy-kitchen` / `sageattention` / `triton`），自动将增量依赖与 ComfyUI 打包为单个 `.tar.zst` 归档存入网盘。
   - **后续断线重连**：直接从网盘顺序读取单个 `.tar.zst` 解压至本地 NVMe，跳过重复 `git clone` 与在线 `pip` 编译，**约 10~15 秒即可拉起网页**。
2. **模型零等待异步内存预热**：
   - 网盘 `ComfyUI_MiniMax_H3/models` 中的 5 件套权重（~70 GiB）秒级软链后，**先启动 ComfyUI 输出网页访问链接**，同时在后台零堆内存开销（`memoryview`）并发预热至 177GB 系统内存缓存。
3. **专属工作流双向持久化**：
   - 自动加载网盘 `ComfyUI_MiniMax_H3/workflows/*.json`（及根目录已有 JSON），且在网页端 `Ctrl+S` 保存的任何修改每 5 秒自动写回网盘，断线重连绝不丢失。
4. **双高速访问通道**：
   - 主线路：Google Colab 官方原生 `proxyPort(8188)` 专线 + `--enable-compress-response-body` Gzip 压缩 + `/userdata/` 路由修复。
   - 备用线路：自动部署 `cloudflared` 并开启强制 HTTP/2 隧道。

---

## 3. 可选环境变量参数

如需强制拉取最新版 ComfyUI 及插件并刷新网盘环境缓存包，可在运行前设置：

```python
import os, urllib.request
os.environ["FORCE_REBUILD_ENV"] = "1"
exec(urllib.request.urlopen("https://raw.githubusercontent.com/nfft894htno32/comfyui-h3-colab/main/run.py").read().decode(), globals())
```
