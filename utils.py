import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 计算评价指标
def compute_metrics(model, dataloader, device):
    all_emotion_preds = []
    all_emotion_labels = []
    all_attribute_preds = []
    all_attribute_labels = []
    
    model.eval()  # 模型切换到评估模式
    with torch.no_grad():  # 关闭梯度计算，节省资源
        for batch in dataloader:
            # 从batch中提取数据并移到指定设备（CPU/GPU）
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].cpu().numpy()
            attribute_labels = batch["attribute_labels"].cpu().numpy()
            
            # 模型推理
            emotion_logits, attribute_logits, _ = model(input_ids, attention_mask)
            # 转换为预测标签（取概率最大的类别）
            emotion_preds = torch.argmax(emotion_logits, dim=1).cpu().numpy()
            attribute_preds = torch.argmax(attribute_logits, dim=1).cpu().numpy()
            
            # 收集所有预测结果和真实标签
            all_emotion_preds.extend(emotion_preds)
            all_emotion_labels.extend(emotion_labels)
            all_attribute_preds.extend(attribute_preds)
            all_attribute_labels.extend(attribute_labels)
    
    # 计算准确率和加权F1值（应对类别不平衡）
    emotion_acc = accuracy_score(all_emotion_labels, all_emotion_preds)
    emotion_f1 = f1_score(all_emotion_labels, all_emotion_preds, average="weighted")
    attribute_acc = accuracy_score(all_attribute_labels, all_attribute_preds)
    attribute_f1 = f1_score(all_attribute_labels, all_attribute_preds, average="weighted")
    
    # 返回指标字典
    return {
        "emotion_acc": emotion_acc, "emotion_f1": emotion_f1,
        "attribute_acc": attribute_acc, "attribute_f1": attribute_f1,
        "emotion_preds": all_emotion_preds, "emotion_labels": all_emotion_labels
    }

# 绘制混淆矩阵
def plot_confusion_matrix(true_labels, pred_labels, save_path):
    # 情感标签映射（0-4对应5个等级）
    labels = ["非常不满", "不满", "中性", "满意", "非常满意"]
    cm = confusion_matrix(true_labels, pred_labels)
    # 设置图表样式
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("情感等级分类混淆矩阵")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # 保存图片到results/figures文件夹
    plt.close()

# 保存指标结果到CSV文件
def save_metrics(metrics, save_path):
    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)