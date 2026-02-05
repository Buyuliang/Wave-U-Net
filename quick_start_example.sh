#!/bin/bash
# 快速开始示例脚本

# 示例 1: 仅使用预训练模型进行预测
echo "示例 1: 使用预训练模型进行预测"
python SingleAudioTrainPredict.py with cfg.full_44KHz \
    input_audio_path="audio_examples/The Mountaineering Club - Mallory/mix.mp3" \
    predict_only=True \
    model_save_path="checkpoints/full_44KHz/full_44KHz-236118"

# 示例 2: 训练单个音频（如果有源文件）
# echo "示例 2: 训练单个音频"
# python SingleAudioTrainPredict.py with cfg.full_44KHz \
#     input_audio_path="your_audio/mix.wav" \
#     vocals_path="your_audio/vocals.wav" \
#     accompaniment_path="your_audio/accompaniment.wav" \
#     num_epochs=10 \
#     steps_per_epoch=100

# 示例 3: 仅使用混合音频训练（不推荐，效果差）
# echo "示例 3: 仅使用混合音频训练"
# python SingleAudioTrainPredict.py with cfg.full_44KHz \
#     input_audio_path="your_audio/mix.wav" \
#     num_epochs=5 \
#     steps_per_epoch=50
