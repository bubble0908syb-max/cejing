"""
测井岩性识别项目 - 时序数据增强模块 (直击内核版 - 完美绕过官方 Bug)
"""

import os
import numpy as np
import pandas as pd

# 🌟 终极核心修复：直接从深层源码导入纯正的 TimeGAN 模型，彻底抛弃有 Bug 的外壳！
from ydata_synthetic.synthesizers.timeseries.timegan.model import TimeGAN
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

def run_timegan_augmentation(pure_data_dir, output_dir, seq_len=15, num_features=8):
    print(f"📥 正在加载纯净 3D 训练集...")
    X_train_pure = np.load(os.path.join(pure_data_dir, 'X_train_pure_3d.npy'))
    y_train_pure = np.load(os.path.join(pure_data_dir, 'y_train_pure.npy'))

    # 统计类别分布，寻找目标数量
    classes, counts = np.unique(y_train_pure, return_counts=True)
    class_distribution = dict(zip(classes, counts))
    max_count = max(counts)

    print(f"\n📊 [纯净训练集] 各类别数量分布: {class_distribution}")
    print(f"🎯 目标平衡数量: {max_count}")

    gan_args = ModelParameters(
        batch_size=32,
        lr=5e-4,
        noise_dim=32,
        layers_dim=64
    )

    train_args = TrainParameters(
        epochs=2, # ⚠️ 测试通过后，务必改回 100 左右慢慢炼丹
        sequence_length=seq_len,
        number_sequences=num_features
    )

    synthetic_X_list = []
    synthetic_y_list = []

    for cls in classes:
        count = class_distribution[cls]
        if count >= max_count:
            continue

        needed_samples = max_count - count
        print(f"\n🚀 准备为少数类别 [{cls}] 训练 TimeGAN，需补齐 {needed_samples} 个序列样本...")

        # 1. 提取当前类的真实 3D 序列
        class_indices = np.where(y_train_pure == cls)[0]
        X_real_class = X_train_pure[class_indices]

        # 直接将 3D 张量拆解为 2D 矩阵列表，这是底层算法最喜欢的纯净格式
        X_real_list = [seq for seq in X_real_class]

        print(f"开始直连底层算法炼丹 (TimeGAN 训练过程较慢，请耐心等待)...")
        # 🌟 直接初始化底层 TimeGAN，参数各司其职，不经过任何多余的校验
        synth = TimeGAN(
            model_parameters=gan_args,
            hidden_dim=24,
            seq_len=seq_len,
            n_seq=num_features,
            gamma=1.0
        )

        # 🌟 直接调用底层的 fit，不传任何无用的 column 参数
        synth.fit(X_real_list, train_args)

        # 生成虚拟序列
        print(f"正在生成 {needed_samples} 个具有物理连续性的虚拟样本...")
        # 底层算法的 sample 参数名是 n_samples
        synthetic_data = synth.sample(n_samples=needed_samples)

        # 将生成的列表直接封装回 3D NumPy 数组
        synthetic_X_list.append(np.array(synthetic_data))
        synthetic_y_list.append(np.full(needed_samples, cls))
        print(f"✅ 类别 [{cls}] 时序增强完毕！")

    # ==========================================
    # 混合：真实 3D 序列 + TimeGAN 虚拟 3D 序列
    # ==========================================
    print("\n" + "="*40)
    print("🧬 正在合并【真实训练数据 + TimeGAN 虚拟训练数据】...")

    if synthetic_X_list:
        X_train_final = np.concatenate([X_train_pure] + synthetic_X_list, axis=0)
        y_train_final = np.concatenate([y_train_pure] + synthetic_y_list, axis=0)
    else:
        X_train_final, y_train_final = X_train_pure, y_train_pure

    # 打乱训练集
    shuffle_idx = np.random.permutation(len(X_train_final))
    X_train_final = X_train_final[shuffle_idx]
    y_train_final = y_train_final[shuffle_idx]

    # 保存给深度学习 (TCN/CNN-LSTM) 用的 3D 训练集
    dl_dir = os.path.join(output_dir, "dl_data")
    os.makedirs(dl_dir, exist_ok=True)
    np.save(os.path.join(dl_dir, 'X_train_dl.npy'), X_train_final)
    np.save(os.path.join(dl_dir, 'y_train_dl.npy'), y_train_final)

    # 拷贝测试集到最终目录以便下游模型统一读取
    np.save(os.path.join(dl_dir, 'X_test_dl.npy'), np.load(os.path.join(pure_data_dir, 'X_test_3d.npy')))
    np.save(os.path.join(dl_dir, 'y_test_dl.npy'), np.load(os.path.join(pure_data_dir, 'y_test.npy')))

    # ==========================================
    # 抽取给树模型 (XGBoost) 用的 2D 训练集
    # ==========================================
    feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    target_col = 'Lith_Encoded'

    X_train_2d = X_train_final[:, -1, :] # 取序列的最后一步
    df_train_2d = pd.DataFrame(X_train_2d, columns=feature_cols)
    df_train_2d[target_col] = y_train_final

    ml_align_dir = os.path.join(output_dir, "ml_aligned_data")
    os.makedirs(ml_align_dir, exist_ok=True)
    df_train_2d.to_csv(os.path.join(ml_align_dir, 'train_aligned.csv'), index=False)

    print(f"\n✅ 终极对齐完毕！")
    print(f"深度学习 3D 数据存至: {dl_dir}")
    print(f"树模型 2D 对齐数据存至: {ml_align_dir}")

if __name__ == "__main__":
    PURE_DATA_DIR = "../data/processed/pure_3d_data"
    OUTPUT_BASE_DIR = "../data/processed"

    try:
        run_timegan_augmentation(PURE_DATA_DIR, OUTPUT_BASE_DIR, seq_len=15)
    except Exception as e:
        import traceback
        print(f"❌ 发生错误: {e}")
        traceback.print_exc()