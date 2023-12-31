@echo off
set PYTHON=.\venv\Scripts\python.exe
set HF_HOME=huggingface
set HF_ENDPOINT=https://hf-mirror.com

chcp 65001 > nul
echo 已启用HuggingFace国内加速...
echo 初次启动需要下载模型，请等待...
chcp 437 > nul

%PYTHON% run_gui.py --lowvram

pause
