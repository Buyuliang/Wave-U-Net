"""
单音频训练和预测脚本
支持对单个音频文件进行训练和预测
"""
# 修复 Python 3.6 兼容性问题：必须在导入其他模块之前执行
import fix_importlib

from sacred import Experiment
from Config import config_ingredient
import tensorflow as tf
import numpy as np
import os
import librosa
import soundfile

import Utils
import Models.UnetAudioSeparator
import Models.UnetSpectrogramSeparator
import Evaluate

ex = Experiment('Single Audio Train and Predict', ingredients=[config_ingredient])

@ex.config
def cfg():
    # 输入音频文件路径（混合音频）
    input_audio_path = None
    
    # 可选的分离源文件路径（用于训练）
    # 如果提供，将用于训练；如果不提供，将使用预训练模型进行预测
    vocals_path = None
    accompaniment_path = None
    
    # 输出路径
    output_path = None  # 如果为None，将在输入文件同目录下创建输出
    
    # 模型保存路径
    model_save_path = None  # 如果为None，将使用默认路径
    
    # 训练参数
    num_epochs = 10  # 训练轮数
    steps_per_epoch = 100  # 每轮训练步数
    learning_rate = 1e-4
    
    # 是否只进行预测（不训练）
    predict_only = False

def create_single_audio_dataset(model_config, input_audio_path, vocals_path=None, accompaniment_path=None):
    """
    从单个音频文件创建训练数据集
    """
    print(f"加载音频文件: {input_audio_path}")
    
    # 加载混合音频
    mix_audio, mix_sr = Utils.load(input_audio_path, sr=model_config["expected_sr"], mono=model_config["mono_downmix"])
    
    # 加载源文件（如果提供）
    sources = {}
    # 根据任务类型加载相应的源文件
    if model_config["task"] == "voice":
        # 人声分离任务
        if vocals_path and os.path.exists(vocals_path):
            vocals_audio, _ = Utils.load(vocals_path, sr=model_config["expected_sr"], mono=model_config["mono_downmix"])
            sources["vocals"] = vocals_audio
            print(f"加载人声文件: {vocals_path}")
        
        if accompaniment_path and os.path.exists(accompaniment_path):
            acc_audio, _ = Utils.load(accompaniment_path, sr=model_config["expected_sr"], mono=model_config["mono_downmix"])
            sources["accompaniment"] = acc_audio
            print(f"加载伴奏文件: {accompaniment_path}")
        
        # 如果没有提供源文件，从混合音频中估计（简单方法：假设人声和伴奏各占一半）
        if "vocals" not in sources or "accompaniment" not in sources:
            print("警告: 未提供完整的分离源文件，将使用混合音频的一半作为估计（仅用于演示）")
            # 这是一个简单的估计，实际应用中需要更好的方法
            if "vocals" not in sources:
                sources["vocals"] = mix_audio * 0.5
            if "accompaniment" not in sources:
                sources["accompaniment"] = mix_audio * 0.5
    elif model_config["task"] == "multi_instrument":
        # 多乐器分离任务 - 需要 bass, drums, other, vocals
        print("警告: 多乐器分离任务需要提供所有源文件（bass, drums, other, vocals）")
        # 这里可以扩展以支持多乐器分离
        raise NotImplementedError("多乐器分离的单音频训练需要提供所有源文件")
    
    # 确保所有音频长度一致
    min_length = min(mix_audio.shape[0], min([s.shape[0] for s in sources.values()]))
    mix_audio = mix_audio[:min_length, :]
    sources = {k: v[:min_length, :] for k, v in sources.items()}
    
    # 确保通道数一致
    if model_config["mono_downmix"]:
        if mix_audio.shape[1] > 1:
            mix_audio = np.mean(mix_audio, axis=1, keepdims=True)
        for k in sources:
            if sources[k].shape[1] > 1:
                sources[k] = np.mean(sources[k], axis=1, keepdims=True)
    else:
        if mix_audio.shape[1] == 1:
            mix_audio = np.tile(mix_audio, [1, 2])
        for k in sources:
            if sources[k].shape[1] == 1:
                sources[k] = np.tile(sources[k], [1, 2])
    
    return mix_audio, sources, mix_sr

def create_tf_dataset_from_audio(mix_audio, sources, model_config, input_shape, output_shape):
    """
    从音频数组创建TensorFlow数据集
    """
    # input_shape 和 output_shape 格式: [batch, time, channels]
    # 但 mix_audio 格式是: [time, channels]
    input_time_frames = input_shape[1]  # 时间维度
    output_time_frames = output_shape[1]  # 时间维度
    num_channels = input_shape[2]  # 通道数
    
    # 计算padding
    pad_frames = (input_time_frames - output_time_frames) // 2
    
    # 确保音频长度足够
    min_required_length = input_time_frames
    if mix_audio.shape[0] < min_required_length:
        # 如果音频太短，进行padding
        padding_needed = min_required_length - mix_audio.shape[0]
        mix_audio = np.pad(mix_audio, [(0, padding_needed), (0, 0)], mode="constant", constant_values=0.0)
        sources = {k: np.pad(v, [(0, padding_needed), (0, 0)], mode="constant", constant_values=0.0) 
                  for k, v in sources.items()}
    
    # 对音频进行padding（用于上下文）
    mix_padded = np.pad(mix_audio, [(pad_frames, pad_frames), (0, 0)], mode="constant", constant_values=0.0)
    sources_padded = {k: np.pad(v, [(pad_frames, pad_frames), (0, 0)], mode="constant", constant_values=0.0) 
                     for k, v in sources.items()}
    
    # 提取片段（使用随机位置，类似原始训练代码）
    snippets = []
    hop_size = output_time_frames  # hop size 等于输出大小
    
    # 确保有足够的片段
    max_start = max(0, mix_audio.shape[0] - input_time_frames)
    num_snippets = model_config.get("num_snippets_per_track", 100)
    
    # 随机选择起始位置
    if max_start > 0:
        start_positions = np.random.randint(0, max_start, size=min(num_snippets, max_start // hop_size + 1))
    else:
        # 如果音频太短，只使用一个片段
        start_positions = [0]
    
    for start_pos in start_positions:
        # 确保不会超出范围
        if start_pos + input_time_frames > mix_padded.shape[0]:
            continue
        
        snippet = {
            "mix": mix_padded[start_pos:start_pos + input_time_frames, :]
        }
        
        # 提取对应的源片段（与 mix 相同大小，后续会用 crop_sample 裁剪）
        for k in sources.keys():
            snippet[k] = sources_padded[k][start_pos:start_pos + input_time_frames, :]
        
        snippets.append(snippet)
    
    # 转换为TensorFlow数据集
    def generator():
        for snippet in snippets:
            yield snippet
    
    output_types = {k: tf.float32 for k in ["mix"] + list(sources.keys())}
    # 所有片段都应该是 input_shape 大小，后续会用 crop_sample 裁剪 source 到输出大小
    output_shapes = {k: tf.TensorShape([input_shape[1], input_shape[2]]) 
                    for k in ["mix"] + list(sources.keys())}
    
    dataset = tf.data.Dataset.from_generator(generator, output_types=output_types, output_shapes=output_shapes)
    
    return dataset

@config_ingredient.capture
def train_single_audio(model_config, input_audio_path, vocals_path, accompaniment_path, 
                       num_epochs, steps_per_epoch, learning_rate, model_save_path):
    """
    训练单个音频文件
    """
    print("=" * 60)
    print("开始单音频训练")
    print("=" * 60)
    
    # 确定输入和输出形状
    disc_input_shape = [model_config["batch_size"], model_config["num_frames"], 0]
    if model_config["network"] == "unet":
        separator_class = Models.UnetAudioSeparator.UnetAudioSeparator(model_config)
    elif model_config["network"] == "unet_spectrogram":
        separator_class = Models.UnetSpectrogramSeparator.UnetSpectrogramSeparator(model_config)
    else:
        raise NotImplementedError
    
    sep_input_shape, sep_output_shape = separator_class.get_padding(np.array(disc_input_shape))
    separator_func = separator_class.get_output
    
    # 加载音频数据
    mix_audio, sources, mix_sr = create_single_audio_dataset(
        model_config, input_audio_path, vocals_path, accompaniment_path
    )
    
    print(f"混合音频形状: {mix_audio.shape}")
    print(f"源音频形状: {[f'{k}: {v.shape}' for k, v in sources.items()]}")
    print(f"输入形状: {sep_input_shape}")
    print(f"输出形状: {sep_output_shape}")
    
    # 创建数据集
    dataset = create_tf_dataset_from_audio(mix_audio, sources, model_config, sep_input_shape, sep_output_shape)
    
    # 应用数据增强（如果启用）
    if model_config["augmentation"]:
        dataset = dataset.map(Utils.random_amplify)
    
    # 裁剪 source 到输出大小（与原始训练代码保持一致）
    crop_frames = (sep_input_shape[1] - sep_output_shape[1]) // 2
    if crop_frames > 0:
        dataset = dataset.map(lambda x: Utils.crop_sample(x, crop_frames))
    
    # 重复、打乱和批处理
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=100)
    # TensorFlow 1.8.0 兼容：使用 apply 方法而不是 batch 的 drop_remainder 参数
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(model_config["batch_size"]))
    dataset = dataset.prefetch(1)
    
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    
    print("构建模型...")
    
    # 构建模型
    separator_sources = separator_func(batch["mix"], True, not model_config["raw_audio_loss"], reuse=False)
    
    # 计算损失
    separator_loss = 0
    for key in model_config["source_names"]:
        if key in batch:
            real_source = batch[key]
            sep_source = separator_sources[key]
            separator_loss += tf.reduce_mean(tf.square(real_source - sep_source))
    separator_loss = separator_loss / float(len([k for k in model_config["source_names"] if k in batch]))
    
    # 设置优化器
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
    increment_global_step = tf.assign(global_step, global_step + 1)
    
    separator_vars = Utils.getTrainableVariables("separator")
    print(f"可训练参数数量: {Utils.getNumParams(separator_vars)}")
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.variable_scope("separator_solver"):
            separator_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
                separator_loss, var_list=separator_vars
            )
    
    # 摘要
    tf.summary.scalar("sep_loss", separator_loss, collections=["sup"])
    sup_summaries = tf.summary.merge_all(key='sup')
    
    # 创建会话
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    # 设置模型保存路径
    if model_save_path is None:
        experiment_id = np.random.randint(0, 1000000)
        model_save_dir = os.path.join(model_config["model_base_dir"], f"single_audio_{experiment_id}")
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"model_{experiment_id}")
    else:
        model_save_dir = os.path.dirname(model_save_path)
        os.makedirs(model_save_dir, exist_ok=True)
    
    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    
    # 训练循环
    print(f"开始训练，共 {num_epochs} 轮，每轮 {steps_per_epoch} 步...")
    _global_step = sess.run(global_step)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for step in range(steps_per_epoch):
            _, loss_val, _global_step = sess.run([separator_solver, separator_loss, increment_global_step])
            epoch_loss += loss_val
            
            if step % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, Loss: {loss_val:.6f}")
        
        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.6f}")
        
        # 每轮保存一次模型
        save_path = saver.save(sess, model_save_path, global_step=int(_global_step))
        print(f"模型已保存到: {save_path}")
    
    print("训练完成！")
    
    # 关闭会话
    sess.close()
    tf.reset_default_graph()
    
    return save_path

@ex.automain
def main(cfg, input_audio_path, vocals_path, accompaniment_path, output_path, 
         model_save_path, num_epochs, steps_per_epoch, learning_rate, predict_only):
    """
    主函数：训练和预测
    """
    model_config = cfg["model_config"]
    
    if input_audio_path is None:
        raise ValueError("必须提供 input_audio_path 参数")
    
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f"输入音频文件不存在: {input_audio_path}")
    
    # 训练模型（如果需要）
    if not predict_only:
        print("\n" + "=" * 60)
        print("步骤 1: 训练模型")
        print("=" * 60)
        trained_model_path = train_single_audio(
            model_config, input_audio_path, vocals_path, accompaniment_path,
            num_epochs, steps_per_epoch, learning_rate, model_save_path
        )
        print(f"\n训练完成！模型保存在: {trained_model_path}")
    else:
        if model_save_path is None:
            raise ValueError("predict_only=True 时，必须提供 model_save_path 参数")
        
        # 检查模型文件是否存在
        import glob
        # TensorFlow checkpoint 文件可能有多个扩展名
        checkpoint_files = glob.glob(model_save_path + "*")
        if not checkpoint_files:
            print("\n" + "=" * 60)
            print("错误: 找不到模型文件！")
            print("=" * 60)
            print(f"模型路径: {model_save_path}")
            print("\n请确保:")
            print("1. 模型文件存在（TensorFlow checkpoint 文件）")
            print("2. 或者下载预训练模型:")
            print("   - 从 https://www.dropbox.com/s/oq0woy3cmf5s8y7/models.zip?dl=1 下载")
            print("   - 解压到 checkpoints/ 目录")
            print("   - 确保路径正确，例如: checkpoints/full_44KHz/full_44KHz-236118")
            print("\n或者，如果您想训练新模型，请设置 predict_only=False")
            raise FileNotFoundError(f"找不到模型文件: {model_save_path}")
        
        trained_model_path = model_save_path
        print(f"使用已训练模型: {trained_model_path}")
    
    # 使用训练好的模型进行预测
    print("\n" + "=" * 60)
    print("步骤 2: 使用模型进行预测")
    print("=" * 60)
    
    # 设置输出路径
    if output_path is None:
        input_dir = os.path.dirname(input_audio_path)
        input_basename = os.path.splitext(os.path.basename(input_audio_path))[0]
        output_path = os.path.join(input_dir, f"{input_basename}_separated")
        os.makedirs(output_path, exist_ok=True)
    
    # 进行预测
    Evaluate.produce_source_estimates(model_config, trained_model_path, input_audio_path, output_path)
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"分离后的音频文件保存在: {output_path}")
