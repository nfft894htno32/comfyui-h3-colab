#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI + MiniMax-H3 (G4 96GB / 8-Step Turbo) Google Colab 一键极速部署与秒级重连恢复脚本
用法 (在 Google Colab 任意代码单元格中运行一行命令):
import urllib.request; exec(urllib.request.urlopen("https://raw.githubusercontent.com/nfft894htno32/comfyui-h3-colab/main/run.py").read().decode(), globals())
"""

import concurrent.futures
import csv
import importlib
import importlib.util
import json
import os
import re
import shutil
import site
import socket
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.request
from pathlib import Path

# ==============================================================================
# 0. 全局配置 (支持通过 os.environ 覆盖，默认零配置即用)
# ==============================================================================
DRIVE_BASE = Path(os.environ.get("DRIVE_BASE_DIR", "/content/drive/MyDrive/ComfyUI_MiniMax_H3"))
DRIVE_MODELS = DRIVE_BASE / "models"
DRIVE_OUTPUTS = DRIVE_BASE / "outputs"
DRIVE_WORKFLOWS = DRIVE_BASE / "workflows"
DRIVE_ENV_CACHE = DRIVE_BASE / "env_cache"

COMFY_DIR = Path("/content/ComfyUI")
CLOUDFLARED_BIN = Path("/usr/local/bin/cloudflared")
FORCE_REBUILD_ENV = os.environ.get("FORCE_REBUILD_ENV", "0") == "1"
SKIP_RAM_PREWARM = os.environ.get("SKIP_RAM_PREWARM", "0") == "1"
SNAPSHOT_SCHEMA = "v1"

os.environ["HF_XET_CACHE"] = "/dev/shm/hf_xet"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CUSTOM_NODES = {
    "ComfyUI-MiniMaxH3-Easy": "https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy.git",
    "ComfyUI-SolAttn_triton": "https://github.com/kijai/ComfyUI-SolAttn_triton.git",
    "TE-Speed-MiniMaxH3": "https://github.com/tl2012tl/TE-Speed-MiniMaxH3.git",
    "Comfyui-Memory_Cleanup": "https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup.git",
    "ComfyUI-Manager": "https://github.com/ltdrdata/ComfyUI-Manager.git",
}

MODEL_MANIFEST = [
    {
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
        "comfy_subdir": "diffusion_models",
        "desc": "扩散主模型 (Pruned BF16 满血无损, 37.46 GiB)",
    },
    {
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "comfy_subdir": "text_encoders",
        "desc": "文本编码器 (Qwen3-VL-32B INT8 ConvRot, 25.28 GiB)",
    },
    {
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "vae/minimax_h3_video_vae_fp16.safetensors",
        "comfy_subdir": "vae",
        "desc": "视频 VAE (FP16, 4.85 GiB)",
    },
    {
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "comfy_subdir": "vae",
        "desc": "音频 VAE (FP32, 0.56 GiB)",
    },
    {
        "repo_id": "Comfy-Org/MiniMax-H3",
        "filename": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "comfy_subdir": "loras",
        "desc": "8步加速 LoRA (Turbo 8-Step BF16, 1.82 GiB)",
    },
]


def log(msg: str) -> None:
    print(msg, flush=True)


# ==============================================================================
# 1. GPU 校验、残留清理与网盘挂载
# ==============================================================================
def init_colab_and_drive() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("❌ 未检测到 GPU！请在 Colab 菜单选择: 代码执行程序 -> 更改运行时类型 -> G4 GPU")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"🖥️ 检测到 GPU: {gpu_name} ({vram_gb:.1f} GB 显存)")

    # 清理可能残留的旧进程，支持在同一会话内重复运行一键重启
    subprocess.run("pkill -9 -f 'main.py --listen'", shell=True, check=False)
    subprocess.run("pkill -9 -f 'python main.py'", shell=True, check=False)
    subprocess.run("pkill -9 -f cloudflared", shell=True, check=False)
    shutil.rmtree("/dev/shm/hf_xet", ignore_errors=True)

    if not Path("/content/drive/MyDrive").exists():
        from google.colab import drive

        log("📂 正在挂载 Google Drive...")
        drive.mount("/content/drive")

    for d in (DRIVE_MODELS, DRIVE_OUTPUTS, DRIVE_WORKFLOWS, DRIVE_ENV_CACHE):
        d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. 环境秒级快照恢复 / 首次并发安装并归档至网盘 (.tar.zst)
# ==============================================================================
def ensure_zstd() -> None:
    if shutil.which("zstd") is None:
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], check=True)


def ensure_cloudflared() -> None:
    if CLOUDFLARED_BIN.exists() and CLOUDFLARED_BIN.stat().st_size > 5 * 1024 * 1024:
        CLOUDFLARED_BIN.chmod(0o755)
        return
    log("🌐 正在下载 cloudflared 备用隧道客户端...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    tmp_bin = Path("/tmp/cloudflared.download")
    urllib.request.urlretrieve(url, str(tmp_bin))
    tmp_bin.chmod(0o755)
    shutil.move(str(tmp_bin), str(CLOUDFLARED_BIN))


def patch_comfyui_userdata_route() -> None:
    """修复 Google Colab 官方代理自动解码 %2F 导致左侧工作流 404 的问题"""
    user_mgr_path = COMFY_DIR / "app" / "user_manager.py"
    if user_mgr_path.exists():
        um_code = user_mgr_path.read_text(encoding="utf-8")
        if '"/userdata/{file}"' in um_code:
            um_code = um_code.replace('"/userdata/{file}"', '"/userdata/{file:.*}"')
            user_mgr_path.write_text(um_code, encoding="utf-8")


def run_pip_fast(args: list[str], check: bool = True) -> None:
    """优先使用 uv 10 倍速安装，若遇到特殊包构建兼容问题自动回退标准 pip"""
    uv_bin = shutil.which("uv")
    if uv_bin:
        res = subprocess.run([uv_bin, "pip", "install", "--system", "-q", *args], check=False)
        if res.returncode == 0:
            return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=check)


def install_all_python_deps() -> None:
    if shutil.which("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=False)

    log("⚡ 正在安装 ComfyUI 核心依赖与 Blackwell 加速库 (comfy-kitchen / SageAttention / Triton)...")
    core_args = [
        "-r",
        str(COMFY_DIR / "requirements.txt"),
        "huggingface_hub[hf_transfer]",
        "comfy-kitchen",
        "sageattention",
        "triton",
        "psutil",
    ]
    run_pip_fast(core_args, check=True)

    custom_nodes_dir = COMFY_DIR / "custom_nodes"
    for name in CUSTOM_NODES:
        req_file = custom_nodes_dir / name / "requirements.txt"
        if req_file.exists():
            run_pip_fast(["-r", str(req_file)], check=False)


def get_all_site_dirs() -> list[Path]:
    candidates = [
        *site.getsitepackages(),
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
    ]
    resolved: set[Path] = set()
    for p_str in candidates:
        if p_str and Path(p_str).exists():
            resolved.add(Path(p_str).resolve())
    return sorted(resolved)


def verify_key_imports() -> bool:
    required_mods = [
        "aiohttp",
        "torchsde",
        "kornia",
        "spandrel",
        "einops",
        "huggingface_hub",
        "comfy_kitchen",
        "sageattention",
        "psutil",
    ]
    return all(importlib.util.find_spec(mod) is not None for mod in required_mods)


def collect_tree_entries(root_path: Path, out_rel_paths: set[str]) -> None:
    """收集目录下所有文件、软链与空目录（跳过 __pycache__），用于精确无重复打包"""
    if not root_path.exists() and not root_path.is_symlink():
        return
    if root_path.is_symlink() or root_path.is_file():
        out_rel_paths.add(str(root_path).lstrip("/"))
        return
    has_children = False
    for child in root_path.iterdir():
        if child.name == "__pycache__":
            continue
        has_children = True
        if child.is_symlink() or child.is_file():
            out_rel_paths.add(str(child).lstrip("/"))
        elif child.is_dir():
            collect_tree_entries(child, out_rel_paths)
    if not has_children:
        out_rel_paths.add(str(root_path).lstrip("/"))


def ensure_comfyui_environment() -> None:
    import torch

    py_ver = f"py{sys.version_info.major}{sys.version_info.minor}"
    torch_ver = torch.__version__.split("+")[0]
    snapshot_name = f"comfyui_g4_env_{py_ver}_torch{torch_ver}_{SNAPSHOT_SCHEMA}.tar.zst"
    snapshot_drive_path = DRIVE_ENV_CACHE / snapshot_name
    ready_marker = COMFY_DIR / f".env_ready_{SNAPSHOT_SCHEMA}"
    replaced_meta = COMFY_DIR / ".replaced_dist_infos.json"
    bin_dir = Path("/usr/local/bin")

    # 情况 A: 当前 Colab 实例已经装好（同会话内重启 ComfyUI）
    if ready_marker.exists() and (COMFY_DIR / "main.py").exists() and not FORCE_REBUILD_ENV:
        ensure_cloudflared()
        patch_comfyui_userdata_route()
        log("✅ 检测到当前实例 ComfyUI 环境已就绪，0 秒直接复用！")
        return

    # 情况 B: 断线重连分配了新机器 -> 从 Google Drive 单文件快照秒级恢复
    if (
        snapshot_drive_path.exists()
        and snapshot_drive_path.stat().st_size > 10 * 1024 * 1024
        and not FORCE_REBUILD_ENV
    ):
        t0 = time.time()
        size_mb = snapshot_drive_path.stat().st_size / (1024**2)
        log(f"🚀 发现网盘离线环境快照 ({snapshot_name}, {size_mb:.0f} MB)，正在极速恢复至本地高速盘...")
        try:
            ensure_zstd()
            local_tar = Path("/tmp/comfyui_env_restore.tar.zst")
            shutil.copyfile(snapshot_drive_path, local_tar)
            subprocess.run(
                [
                    "tar",
                    "--use-compress-program=zstd",
                    "--keep-directory-symlink",
                    "--overwrite",
                    "-xf",
                    str(local_tar),
                    "-C",
                    "/",
                ],
                check=True,
            )
            local_tar.unlink(missing_ok=True)

            # 清理被升级替换掉的旧版 .dist-info，保持包元数据干净
            if replaced_meta.exists():
                replaced_map: dict[str, list[str]] = json.loads(replaced_meta.read_text(encoding="utf-8"))
                for sdir_str, old_dis in replaced_map.items():
                    sdir_path = Path(sdir_str)
                    for old_di in old_dis:
                        old_path = sdir_path / old_di
                        if old_path.exists():
                            shutil.rmtree(old_path, ignore_errors=True)

            importlib.invalidate_caches()
            ensure_cloudflared()
            patch_comfyui_userdata_route()

            if not verify_key_imports():
                log("⚠️ 检测到 Colab 基础镜像微调，正在增量补齐缺失依赖...")
                install_all_python_deps()

            ready_marker.write_text("ok", encoding="utf-8")
            log(f"✅ 环境从网盘快照秒级恢复完成！(耗时 {time.time() - t0:.1f}s)")
            return
        except Exception as exc:
            log(f"⚠️ 快照恢复异常 ({exc})，自动回退至在线构建并刷新快照...")
            snapshot_drive_path.unlink(missing_ok=True)

    # 情况 C: 首次运行（或手动指定 FORCE_REBUILD_ENV=1）-> 并发克隆 + uv 极速安装 + 生成网盘快照
    t0 = time.time()
    log("📦 首次初始化环境：正在并发克隆 ComfyUI 与 5 个核心自定义节点...")
    if FORCE_REBUILD_ENV and COMFY_DIR.exists():
        shutil.rmtree(COMFY_DIR, ignore_errors=True)

    if not (COMFY_DIR / "main.py").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_DIR)],
            check=True,
        )

    custom_nodes_dir = COMFY_DIR / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)

    def _clone_node(item: tuple[str, str]) -> None:
        name, url = item
        target = custom_nodes_dir / name
        if not target.exists():
            subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(_clone_node, CUSTOM_NODES.items()))

    ensure_cloudflared()

    # 记录安装前所有 site-packages 目录与 bin 快照，仅截取增量包打包进缓存（避免把 Colab 自带的数 GB PyTorch 重复打包）
    site_dirs_before = get_all_site_dirs()
    before_top_map = {
        sdir: {p.name: p.stat().st_mtime_ns for p in sdir.iterdir()} for sdir in site_dirs_before
    }
    before_di_map = {
        sdir: {p.name for p in sdir.glob("*.dist-info")} for sdir in site_dirs_before
    }
    before_bins = {p.name: p.stat().st_mtime_ns for p in bin_dir.iterdir()} if bin_dir.exists() else {}

    install_all_python_deps()
    patch_comfyui_userdata_route()

    site_dirs_after = get_all_site_dirs()
    replaced_map_out: dict[str, list[str]] = {}
    new_di_map: dict[Path, set[str]] = {}

    for sdir in site_dirs_after:
        b_dis = before_di_map.get(sdir, set())
        a_dis = {p.name for p in sdir.glob("*.dist-info")}
        rep = sorted(b_dis - a_dis)
        if rep:
            replaced_map_out[str(sdir)] = rep
        new_di_map[sdir] = a_dis - b_dis

    replaced_meta.write_text(json.dumps(replaced_map_out), encoding="utf-8")
    ready_marker.write_text("ok", encoding="utf-8")

    # 打包 /content/ComfyUI + cloudflared + 新增 Python 依赖增量至 Google Drive，供后续每次重连秒级恢复
    try:
        ensure_zstd()
        log("💾 正在将当前完整环境打包为单个 .tar.zst 快照存入网盘（后续每次断线重连仅需 ~10 秒恢复）...")
        rel_files: set[str] = set()
        collect_tree_entries(COMFY_DIR, rel_files)
        collect_tree_entries(CLOUDFLARED_BIN, rel_files)

        for sdir in site_dirs_after:
            b_top = before_top_map.get(sdir, {})
            for p in sdir.iterdir():
                if p.name == "__pycache__":
                    continue
                if p.name not in b_top or p.stat().st_mtime_ns != b_top[p.name]:
                    collect_tree_entries(p, rel_files)

            for di_name in new_di_map.get(sdir, set()):
                rec_file = sdir / di_name / "RECORD"
                if rec_file.exists():
                    with open(rec_file, "r", encoding="utf-8", errors="ignore") as f:
                        for row in csv.reader(f):
                            if not row or not row[0]:
                                continue
                            abs_p = (sdir / row[0]).resolve()
                            if (abs_p.is_file() or abs_p.is_symlink()) and "__pycache__" not in abs_p.parts:
                                rel_files.add(str(abs_p).lstrip("/"))

        if bin_dir.exists():
            for b in bin_dir.iterdir():
                if b.name not in before_bins or b.stat().st_mtime_ns != before_bins[b.name]:
                    collect_tree_entries(b, rel_files)

        lst_file = Path("/tmp/comfy_snapshot_files.lst")
        local_tar = Path("/tmp/comfy_snapshot_build.tar.zst")
        lst_file.write_bytes(b"\0".join(p.encode("utf-8") for p in sorted(rel_files)) + b"\0")

        zstd_env = {**os.environ, "ZSTD_NBTHREADS": "0", "ZSTD_CLEVEL": "3"}
        subprocess.run(
            [
                "tar",
                "--use-compress-program=zstd",
                "--no-recursion",
                "--null",
                "-T",
                str(lst_file),
                "-cf",
                str(local_tar),
                "-C",
                "/",
            ],
            env=zstd_env,
            check=True,
        )
        part_path = DRIVE_ENV_CACHE / (snapshot_name + ".part")
        shutil.copyfile(local_tar, part_path)
        part_path.replace(snapshot_drive_path)
        lst_file.unlink(missing_ok=True)
        local_tar.unlink(missing_ok=True)
        saved_mb = snapshot_drive_path.stat().st_size / (1024**2)
        log(f"✅ 环境构建并归档网盘完成 ({saved_mb:.0f} MB, 总耗时 {time.time() - t0:.1f}s)！下次重连将秒级恢复。")
    except Exception as exc:
        log(f"⚠️ 网盘快照归档跳过 ({exc})，不影响本次正常启动。")


# ==============================================================================
# 3. 挂载网盘已有的 5 件套模型、输出目录与持久化工作流
# ==============================================================================
def link_models_and_outputs() -> list[Path]:
    from huggingface_hub import hf_hub_download

    drive_model_files: list[Path] = []
    for idx, item in enumerate(MODEL_MANIFEST, 1):
        basename = Path(item["filename"]).name
        drive_file = DRIVE_MODELS / item["filename"]
        comfy_target_dir = COMFY_DIR / "models" / item["comfy_subdir"]
        comfy_target_dir.mkdir(parents=True, exist_ok=True)
        comfy_link = comfy_target_dir / basename

        if drive_file.exists() and drive_file.stat().st_size > 1024 * 1024:
            size_gib = drive_file.stat().st_size / (1024**3)
            log(f"✅ [{idx}/{len(MODEL_MANIFEST)}] 网盘模型就绪: {basename} ({size_gib:.2f} GiB)")
        else:
            log(f"📥 [{idx}/{len(MODEL_MANIFEST)}] 网盘未找到，正在下载 {item['desc']} -> {basename} ...")
            hf_hub_download(
                repo_id=item["repo_id"],
                filename=item["filename"],
                local_dir=str(DRIVE_MODELS),
            )
            shutil.rmtree("/dev/shm/hf_xet", ignore_errors=True)

        if comfy_link.exists() or comfy_link.is_symlink():
            comfy_link.unlink()
        os.symlink(str(drive_file), str(comfy_link))
        drive_model_files.append(drive_file)

    comfy_output = COMFY_DIR / "output"
    if comfy_output.exists() and not comfy_output.is_symlink():
        shutil.rmtree(comfy_output, ignore_errors=True)
    if not comfy_output.exists():
        os.symlink(str(DRIVE_OUTPUTS), str(comfy_output))

    return drive_model_files


def setup_and_sync_workflows() -> None:
    """
    优先加载你网盘中已有的私密工作流 JSON（如 MiniMax_H3_G4_8Step_漫画动起来.json 或 workflows/*.json），
    并在后台实时将你在网页端 Ctrl+S 保存的工作流自动同步回网盘；
    仅当网盘完全没有任何工作流时，才用中性占位提示词生成默认 8 步极速模板。
    """
    user_wf_dir = COMFY_DIR / "user" / "default" / "workflows"
    user_wf_dir.mkdir(parents=True, exist_ok=True)

    # 自动迁移旧版保存在 ComfyUI_MiniMax_H3 根目录下的工作流 JSON 到 workflows/ 子目录
    for root_json in DRIVE_BASE.glob("*.json"):
        target_in_wf = DRIVE_WORKFLOWS / root_json.name
        if not target_in_wf.exists():
            shutil.copy2(root_json, target_in_wf)

    existing_drive_wfs = sorted(DRIVE_WORKFLOWS.glob("*.json"))
    if existing_drive_wfs:
        for wf_file in existing_drive_wfs:
            shutil.copy2(wf_file, user_wf_dir / wf_file.name)
        names_str = ", ".join(p.name for p in existing_drive_wfs)
        log(f"✅ 已从网盘加载你的专属工作流 ({len(existing_drive_wfs)} 个): {names_str}")
    else:
        easy_wf_src = COMFY_DIR / "custom_nodes" / "ComfyUI-MiniMaxH3-Easy" / "workflow" / "1.MiniMax_H3_Easy.json"
        if easy_wf_src.exists():
            wf = json.loads(easy_wf_src.read_text(encoding="utf-8"))
            neutral_prompt = (
                "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video.\n\n"
                "integrated_multimodal_description: [Shot 1] 固定三脚架机位，画面构图全程保持静止，镜头不推拉、不摇移。画面中的动漫角色自然站在户外场景中 <Picture 1>，微风轻轻拂动发丝与衣角，眼神自然眨动并开口说话，画面文字框保持静止不动。角色用自然清晰的中文说道：<d> \"你好，欢迎来到动画测试场景。\" </d>\n"
                "overall_soundscape: 轻柔的户外环境音、细微的衣物摩擦声，以及自然清晰的中文配音。\n"
                "non_diegetic_music: N/A"
            )
            for node in wf.get("nodes", []):
                ntype = node.get("type")
                if ntype == "MiniMaxH3EasyLoader":
                    node["widgets_values"] = [
                        "minimax_h3_fl2va_pruned_bf16.safetensors",
                        "无",
                        "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                        "minimax_h3_video_vae_fp16.safetensors",
                        "minimax_h3_audio_vae_fp32.safetensors",
                    ]
                elif ntype == "LoraLoaderModelOnly":
                    node["widgets_values"] = [
                        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                        1.0,
                    ]
                elif ntype == "BasicScheduler":
                    node["widgets_values"] = ["simple", 8, 1.0]
                elif ntype == "ModelAttentionBackend":
                    node["widgets_values"] = ["comfy kitchen attention"]
                elif ntype == "RAMCleanup":
                    node["widgets_values"] = [False, False, False, 1]
                    node["mode"] = 4
                elif ntype == "MiniMaxH3Easy":
                    node["widgets_values"] = [
                        "图生或首尾帧",
                        neutral_prompt,
                        None,
                        "768P",
                        "2:3",
                        832,
                        1216,
                        5,
                        False,
                        24,
                        "首帧优先",
                        "原图（不缩放）",
                        "按序号",
                        False,
                        "none",
                    ]
                    if "properties" in node:
                        if "minimax_h3_prompt_optimizer" in node["properties"]:
                            node["properties"]["minimax_h3_prompt_optimizer"]["enabled"] = False
                        node["properties"]["minimax_h3_prompt_reference_doc"] = {
                            "version": 1,
                            "text": neutral_prompt,
                            "parts": [{"type": "text", "text": neutral_prompt}],
                        }

            default_name = "MiniMax_H3_G4_8Step_Default.json"
            wf_text = json.dumps(wf, ensure_ascii=False, indent=2)
            (user_wf_dir / default_name).write_text(wf_text, encoding="utf-8")
            (DRIVE_WORKFLOWS / default_name).write_text(wf_text, encoding="utf-8")
            log(f"✅ 已初始化默认 8 步极速工作流模板: {default_name}")

    # 启动后台轻量同步线程：你在 ComfyUI 网页端保存的任何工作流修改，每 5 秒自动持久化回 Google Drive
    synced_mtimes: dict[str, int] = {
        p.name: p.stat().st_mtime_ns for p in user_wf_dir.glob("*.json")
    }

    def _watch_workflows() -> None:
        while True:
            time.sleep(5)
            try:
                for p in user_wf_dir.glob("*.json"):
                    mtime = p.stat().st_mtime_ns
                    if synced_mtimes.get(p.name) != mtime:
                        shutil.copy2(p, DRIVE_WORKFLOWS / p.name)
                        synced_mtimes[p.name] = mtime
            except Exception:
                pass

    threading.Thread(target=_watch_workflows, daemon=True).start()


# ==============================================================================
# 4. 后台非阻塞 70 GiB 模型内存缓存预热 (不卡网页启动，零内存分配读取)
# ==============================================================================
def start_background_ram_prewarm(model_files: list[Path]) -> None:
    if SKIP_RAM_PREWARM:
        return

    def _warm_one(fpath: Path) -> int:
        read_bytes = 0
        try:
            fd = os.open(str(fpath), os.O_RDONLY)
            try:
                if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_WILLNEED"):
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            finally:
                os.close(fd)
        except Exception:
            pass

        buf = bytearray(64 * 1024 * 1024)
        mv = memoryview(buf)
        with open(fpath, "rb", buffering=0) as fp:
            while True:
                n = fp.readinto(mv)
                if not n:
                    break
                read_bytes += n
        return read_bytes

    def _runner() -> None:
        t0 = time.time()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                total_b = sum(pool.map(_warm_one, model_files))
            gib = total_b / (1024**3)
            log(f"🔥 [后台预热完成] 全部 {len(model_files)} 个模型 ({gib:.1f} GiB) 已常驻系统内存缓存 (耗时 {time.time() - t0:.1f}s)！")
        except Exception as exc:
            log(f"⚠️ 后台模型预热提示: {exc}")

    log("🚀 已在后台启动 70 GiB 模型内存预热（不阻塞网页启动，你打开网页选图期间即可自动完成）...")
    threading.Thread(target=_runner, daemon=True).start()


# ==============================================================================
# 5. 启动官方直连专线 + Cloudflare HTTP/2 备用隧道 + ComfyUI 主进程
# ==============================================================================
def launch_comfyui() -> None:
    colab_proxy_url = None
    try:
        from google.colab import output

        colab_proxy_url = output.eval_js(
            "(() => {"
            "  try {"
            "    if (navigator.locks && !window.__comfyKeepAliveLock) {"
            "      window.__comfyKeepAliveLock = true;"
            "      navigator.locks.request('colab_comfyui_keepalive', () => new Promise(() => {}));"
            "    }"
            "  } catch (e) {}"
            "  return google.colab.kernel.proxyPort(8188);"
            "})()"
        )
    except Exception:
        pass

    def _start_tunnels() -> None:
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", 8188)) == 0:
                    break
            time.sleep(0.5)

        log("\n" + "=" * 84)
        if colab_proxy_url:
            log("🚀 【高速主线路 - 强烈推荐】Google Colab 官方直连专线（与当前网页同速，绝不卡顿）：")
            log(f"👉 {colab_proxy_url}")
            log("=" * 84)

        if CLOUDFLARED_BIN.exists():
            try:
                proc = subprocess.Popen(
                    [
                        str(CLOUDFLARED_BIN),
                        "tunnel",
                        "--url",
                        "http://127.0.0.1:8188",
                        "--protocol",
                        "http2",
                        "--no-autoupdate",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                    if match:
                        log(f"🌐 【备用线路】Cloudflare HTTP/2 隧道：{match.group(0)}")
                        log("=" * 84 + "\n")
                        break
            except Exception as exc:
                log(f"⚠️ Cloudflare 备用隧道启动跳过: {exc}")

    threading.Thread(target=_start_tunnels, daemon=True).start()

    log("🎬 正在启动高速版 ComfyUI (--highvram + Gzip 响应压缩)...")
    try:
        subprocess.run(
            [
                sys.executable,
                "main.py",
                "--listen",
                "0.0.0.0",
                "--port",
                "8188",
                "--enable-cors-header",
                "--enable-compress-response-body",
                "--highvram",
            ],
            cwd=str(COMFY_DIR),
            check=True,
        )
    except KeyboardInterrupt:
        log("\n🛑 ComfyUI 服务已手动停止。")


def main() -> None:
    t_boot = time.time()
    init_colab_and_drive()
    ensure_comfyui_environment()
    model_files = link_models_and_outputs()
    setup_and_sync_workflows()
    log(f"⚡ 前置部署总耗时仅 {time.time() - t_boot:.1f}s，即将拉起 ComfyUI 服务...")
    start_background_ram_prewarm(model_files)
    launch_comfyui()


if os.environ.get("COMFY_IMPORT_ONLY") != "1":
    main()
