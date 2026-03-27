"""
测井岩性识别项目 - Vision Transformer (ViT-1D) 模型
状态：动态特征筛选 + 5 分类 + 全局自注意力机制
加固：引入 CLS Token + 可学习位置编码 + AdamW 优化器 + GELU 激活
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ==========================================
# 1. 定义 1D Vision Transformer 架构
# ==========================================
class ViT1D(nn.Module):
    def __init__(self, seq_len, num_features, num_classes, d_model=64, nhead=4, num_layers=3, dim_feedforward=128,
                 dropout=0.3):
        super(ViT1D, self).__init__()

        # 1. 线性投影层：将原始特征维度映射到 Transformer 的 d_model 维度
        self.feature_projection = nn.Linear(num_features, d_model)

        # 2. CLS Token 和位置编码 (Positional Embedding)
        # 序列长度加 1 是因为引入了 CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder 核心层
        # ViT 标配通常使用 GELU 激活函数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. LayerNorm 与 分类头 (MLP Head)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, num_features)
        """
        b = x.shape[0]

        # 1. 特征投影 -> (batch_size, seq_len, d_model)
        x = self.feature_projection(x)

        # 2. 拼接 CLS Token -> (batch_size, seq_len + 1, d_model)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 3. 加入位置编码
        x = x + self.pos_embedding
        x = self.pos_drop(x)

        # 4. 通过 Transformer Encoder
        x = self.transformer_encoder(x)

        # 5. 提取 CLS Token 对应的输出进行分类 (索引 0 的位置)
        cls_out = x[:, 0, :]
        cls_out = self.norm(cls_out)

        # 6. 分类输出
        logits = self.classifier(cls_out)
        return logits


# ==========================================
# 2. 辅助绘图函数
# ==========================================
def plot_training_curves(train_losses, test_f1_scores, output_dir, filename='vit_1d_training_curves.png'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('ViT-1D Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_f1_scores, 'r-', label='Test Macro-F1')
    plt.title('ViT-1D Test Macro-F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Macro-F1')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)


# ==========================================
# 3. 训练与评估流程
# ==========================================
def train_and_evaluate_vit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在使用设备: {device}")

    # 加载数据
    data_dir = "../data/processed/dl_data"
    X_train = np.load(os.path.join(data_dir, 'X_train_dl.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_dl.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test_dl.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test_dl.npy'))

    # 动态特征过滤
    original_feature_cols = ['_CAL', '_GR', '_SP', '_LLD', '_LLS', '_AC', '_DEN', '_PEF']
    selected_features_path = "../reports/figures/selected_features.txt"

    if os.path.exists(selected_features_path):
        with open(selected_features_path, 'r') as f:
            selected_features = [line.strip() for line in f.readlines() if line.strip()]
        selected_indices = [original_feature_cols.index(feat) for feat in selected_features if
                            feat in original_feature_cols]
        X_train = X_train[:, :, selected_indices]
        X_test = X_test[:, :, selected_indices]
        print(f"✅ 使用特征: {selected_features}")

    # 获取类别名称
    try:
        le = joblib.load('../data/processed/ml_data/label_encoder.pkl')
        class_names = list(le.classes_)
    except:
        class_names = [f'Class {i}' for i in range(5)]

    # 类别权重
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    seq_len = X_train.shape[1]
    num_features = X_train.shape[2]
    num_classes = len(np.unique(y_train))

    print(f"📐 模型输入: 序列长度={seq_len}, 特征数={num_features}, 输出类别数={num_classes}")

    # 初始化 ViT 模型
    # d_model=64, nhead=4, num_layers=3 是适合较小时间序列数据集的轻量化配置
    model = ViT1D(seq_len=seq_len, num_features=num_features, num_classes=num_classes,
                  d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # 🌟 针对 Transformer，使用 AdamW 通常比 Adam 效果更好，且引入 Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)

    # 引入调度器控制学习率衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    epochs = 100
    print(f"\n🔥 开始训练 Vision Transformer (ViT-1D) 模型...")

    best_f1 = 0.0
    model_save_dir = "../saved_models/deep_learning"
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = os.path.join(model_save_dir, "vit_1d_5class.pth")

    history_train_loss = []
    history_test_f1 = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 🌟 Transformer 模型非常需要梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # 验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        current_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        history_test_f1.append(current_macro_f1)

        scheduler.step(avg_test_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_acc = accuracy_score(all_labels, all_preds)
            print(
                f"Epoch [{epoch + 1:03d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test F1: {current_macro_f1:.4f}")

        if current_macro_f1 > best_f1:
            best_f1 = current_macro_f1
            torch.save(model.state_dict(), best_model_path)

    print(f"\n✅ 训练完毕！ViT-1D 最佳 Macro-F1 分数为: {best_f1:.4f}")

    # ==========================================
    # 4. 最终评估与保存图表
    # ==========================================
    report_dir = "../reports/figures"
    plot_training_curves(history_train_loss, history_test_f1, report_dir, filename='vit_1d_training_curves.png')

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    final_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())

    print("\n[ViT-1D 架构] 终极分类报告:")
    print("=" * 60)
    report_str = classification_report(y_test, final_preds, target_names=class_names)
    print(report_str)

    final_acc = accuracy_score(y_test, final_preds)

    # 保存文字报告
    with open(os.path.join(report_dir, 'vit_1d_report.txt'), 'w', encoding='utf-8') as f:
        f.write("[ViT-1D 架构] 分类报告:\n" + "=" * 60 + "\n" + report_str)
        f.write(f"\n🎯 最终 Accuracy 分数: {final_acc:.4f}\n")
        f.write(f"🎯 最终 Macro-F1 分数: {best_f1:.4f}\n")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'ViT-1D Confusion Matrix (F1: {best_f1:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "vit_1d_confusion_matrix.png"), dpi=300)
    print(f"📊 混淆矩阵与文字报告已保存至 {report_dir}")


if __name__ == "__main__":
    try:
        train_and_evaluate_vit()
    except Exception as e:
        print(f"❌ 发生错误: {e}")