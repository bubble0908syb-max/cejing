"""
测井岩性识别项目 - 真实时序切分与防泄露划分模块 (TimeGAN 前置准备)
职责：直接读取预处理后的纯净数据，切出真实的 3D 序列，划分 Train/Test 并隔离测试集。
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_sequences_by_well(df, sequence_length, feature_cols, target_col):
    """基于单口井的真实物理深度进行滑动窗口切分"""
    X_list, y_list = [], []
    df = df.sort_values(by=['Well_Name', 'TopDepth']).reset_index(drop=True)
    grouped = df.groupby('Well_Name')

    for well_name, group in grouped:
        group_X = group[feature_cols].values
        group_y = group[target_col].values

        if len(group) < sequence_length:
            continue

        for i in range(len(group) - sequence_length + 1):
            window_X = group_X[i: i + sequence_length, :]
            # 标签取序列最后一个时间步的岩性
            window_y = group_y[i + sequence_length - 1]

            X_list.append(window_X)
            y_list.append(window_y)

    return np.array(X_list), np.array(y_list)


def process_pure_sequences(input_filepath, output_dir, sequence_length=15):
    print(f"📥 正在加载全局干净预处理数据: {input_filepath}")
    df = pd.read_csv(input_filepath)

    feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    target_col = 'Lith_Encoded'

    # 1. 纯净 3D 序列构建
    print(f"🔄 正在基于真实井深构建 {sequence_length} 步长的物理连续序列...")
    X_3d, y_3d = create_sequences_by_well(df, sequence_length, feature_cols, target_col)
    print(f"📊 共切分出 {len(X_3d)} 个真实序列样本。")

    # 2. 核心：在任何增强之前，先切分并隔离 Test 集！
    print("🛡️ 正在进行防泄露的 Train/Test 划分 (Test 集将被永久隔离)...")
    X_train_pure, X_test, y_train_pure, y_test = train_test_split(
        X_3d, y_3d, test_size=0.2, random_state=42, stratify=y_3d
    )

    # 3. 保存纯净数据供后续 TimeGAN 使用
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train_pure_3d.npy'), X_train_pure)
    np.save(os.path.join(output_dir, 'y_train_pure.npy'), y_train_pure)
    np.save(os.path.join(output_dir, 'X_test_3d.npy'), X_test)  # 测试集直接定稿，不再修改
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    # 同步抽取 Test 集的 2D 数据给树模型备用
    X_test_2d = X_test[:, -1, :]
    df_test_2d = pd.DataFrame(X_test_2d, columns=feature_cols)
    df_test_2d[target_col] = y_test
    df_test_2d.to_csv(os.path.join(output_dir, 'test_aligned.csv'), index=False)

    print(f"✅ 纯净序列切分完毕！已保存至: {output_dir}")
    print(f"接下来请运行 TimeGAN 增强模块对 X_train_pure_3d.npy 进行扩充。")


if __name__ == "__main__":
    # 注意：这里读取的是 preprocessed_data.csv，而不是之前 CTGAN 生成的！
    INPUT_FILE = "../data/processed/preprocessed_data.csv"
    PURE_DATA_DIR = "../data/processed/pure_3d_data"

    try:
        process_pure_sequences(INPUT_FILE, PURE_DATA_DIR, sequence_length=15)
    except Exception as e:
        print(f"❌ 发生错误: {e}")