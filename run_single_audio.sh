#!/bin/bash
# 单音频训练和预测启动脚本
# 自动使用 conda 环境的 Python

# 设置 conda 环境的 Python 路径
PYTHON_BIN="/opt/anaconda3/envs/wave-unet/bin/python"

# 检查 Python 是否存在
if [ ! -f "$PYTHON_BIN" ]; then
    echo "错误: 找不到 conda 环境的 Python: $PYTHON_BIN"
    echo "请确保 wave-unet conda 环境已创建"
    exit 1
fi

# 检查依赖是否安装
echo "检查依赖..."
$PYTHON_BIN -c "import sacred, librosa, tensorflow, numba" 2>/dev/null || {
    echo "警告: 缺少依赖，正在安装..."
    echo "这可能需要几分钟时间..."
    $PYTHON_BIN -m pip install -r requirements.txt
    echo "依赖安装完成！"
}

# 运行脚本
echo "使用 Python: $PYTHON_BIN"
echo "Python 版本: $($PYTHON_BIN --version)"
echo ""

# 执行传入的所有参数
$PYTHON_BIN SingleAudioTrainPredict.py "$@"



./run_single_audio.sh with cfg.full_44KHz \
    input_audio_path="nobobby/nobobby_mix.wav" \
    vocals_path="nobobby/nobobby_vocals.wav" \
    accompaniment_path="nobobby/nobobby_accompaniment.wav" \
    num_epochs=10 \
    steps_per_epoch=100 \
    predict_only=False