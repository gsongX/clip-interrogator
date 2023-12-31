
$path = ".\input" # 图片存在路径，默认为目录下的input文件夹，不要有中文
$vram = 0 # 显存如果低于8g，改为1

.\venv\Scripts\activate
$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT="https://hf-mirror.com"
$ext_args = @()

Write-Output "已启用HuggingFace国内加速..."
Write-Output "初次启动需要下载模型，请等待..."

if ($vram) {
    $ext_args += "--lowvram"
}

python run_fast.py --i_path $path $ext_args

pause
