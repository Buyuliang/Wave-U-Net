# 单音频训练和预测使用说明

## 概述

`SingleAudioTrainPredict.py` 脚本允许您对单个音频文件进行训练和预测，无需使用完整的 MUSDB 数据集。

## 功能特点

1. **单音频训练**：使用单个音频文件（混合音频 + 可选的分离源）进行模型训练
2. **自动预测**：训练完成后自动使用训练好的模型进行预测
3. **灵活配置**：支持多种配置选项和参数调整

## 使用方法

### 基本用法（仅预测，使用预训练模型）

如果您只想使用预训练模型进行预测，不进行训练：

```bash
python SingleAudioTrainPredict.py with cfg.full_44KHz \
    input_audio_path="path/to/your/audio.mp3" \
    predict_only=True \
    model_save_path="checkpoints/full_44KHz/full_44KHz-236118"
```

### 完整流程（训练 + 预测）

如果您有混合音频和分离的源文件（人声和伴奏），可以进行训练：

```bash
python SingleAudioTrainPredict.py with cfg.full_44KHz \
    input_audio_path="path/to/mix.mp3" \
    vocals_path="path/to/vocals.wav" \
    accompaniment_path="path/to/accompaniment.wav" \
    num_epochs=10 \
    steps_per_epoch=100 \
    learning_rate=1e-4
```

### 仅使用混合音频训练（不推荐）

如果没有分离的源文件，脚本会使用简单的估计方法（将混合音频分成两半），但效果会很差：

```bash
python SingleAudioTrainPredict.py with cfg.full_44KHz \
    input_audio_path="path/to/mix.mp3" \
    num_epochs=5 \
    steps_per_epoch=50
```

## 参数说明

### 必需参数

- `input_audio_path`: 输入音频文件路径（混合音频）

### 可选参数

#### 训练相关
- `vocals_path`: 人声文件路径（用于训练）
- `accompaniment_path`: 伴奏文件路径（用于训练）
- `num_epochs`: 训练轮数（默认：10）
- `steps_per_epoch`: 每轮训练步数（默认：100）
- `learning_rate`: 学习率（默认：1e-4）
- `model_save_path`: 模型保存路径（默认：自动生成）

#### 输出相关
- `output_path`: 输出目录路径（默认：输入文件同目录下的 `{filename}_separated` 文件夹）
- `predict_only`: 是否只进行预测，不训练（默认：False）

## 配置选项

脚本支持使用 Config.py 中定义的所有配置，例如：

- `cfg.full_44KHz`: 44.1 KHz 采样率的完整模型（推荐用于人声分离）
- `cfg.full`: 22.05 KHz 采样率的完整模型
- `cfg.baseline_stereo`: 基础立体声模型
- `cfg.full_multi_instrument`: 多乐器分离模型

## 输出文件

预测完成后，分离后的音频文件将保存在输出目录中，文件名格式为：
- `{原文件名}_vocals.wav`
- `{原文件名}_accompaniment.wav`

## 注意事项

1. **训练数据质量**：如果提供分离的源文件，训练效果会更好。没有源文件时，训练效果会很差。

2. **训练时间**：单音频训练通常比完整数据集训练快得多，但模型可能过拟合到单个音频。

3. **模型配置**：确保使用的配置（如 `cfg.full_44KHz`）与您的需求匹配。

4. **音频格式**：支持常见的音频格式（mp3, wav, flac 等），librosa 会自动处理。

5. **采样率**：音频会自动重采样到模型配置中指定的采样率。

## 示例

### 示例 1：使用项目中的示例音频

```bash
python SingleAudioTrainPredict.py with cfg.full_44KHz \
    input_audio_path="audio_examples/The Mountaineering Club - Mallory/mix.mp3" \
    predict_only=True \
    model_save_path="checkpoints/full_44KHz/full_44KHz-236118"
```

### 示例 2：训练自己的音频（有源文件）

```bash
python SingleAudioTrainPredict.py with cfg.full_44KHz \
    input_audio_path="my_audio/mix.wav" \
    vocals_path="my_audio/vocals.wav" \
    accompaniment_path="my_audio/accompaniment.wav" \
    num_epochs=20 \
    steps_per_epoch=200 \
    output_path="my_audio/separated"
```

## 工作流程

1. **加载音频**：加载混合音频和可选的源文件
2. **预处理**：重采样、通道处理、padding
3. **创建数据集**：从音频中提取训练片段
4. **训练模型**：使用 TensorFlow 训练 Wave-U-Net 模型
5. **保存模型**：将训练好的模型保存到指定路径
6. **预测**：使用训练好的模型对输入音频进行分离
7. **保存结果**：将分离后的音频保存为 WAV 文件

## 故障排除

### 问题：内存不足
- 减少 `batch_size`（在 Config.py 中）
- 减少 `steps_per_epoch`
- 使用更短的音频文件

### 问题：训练损失不下降
- 检查源文件是否正确
- 增加训练轮数
- 调整学习率

### 问题：预测结果不理想
- 确保使用了正确的模型配置
- 如果只训练了单个音频，模型可能过拟合
- 尝试使用预训练模型进行预测
