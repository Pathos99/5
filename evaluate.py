import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertSentimentAttributeModel
from data_preprocess import CommentDataset
from utils import compute_metrics, plot_confusion_matrix, save_metrics
import os

# 评测参数配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LEN = 128
MODEL_PATH = "./results/model_weights/best_model.pth"  # 训练好的模型路径
TEST_DATA_PATH = "./data/processed_data/test.csv"
METRICS_SAVE_PATH = "./results/metrics/test_metrics.csv"
CONFUSION_MATRIX_PATH = "./results/figures/confusion_matrix.png"

def evaluate_model():
    # 1. 创建文件夹（若不存在）
    os.makedirs("./results/metrics", exist_ok=True)
    os.makedirs("./results/figures", exist_ok=True)
    
    # 2. 初始化分词器和模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertSentimentAttributeModel().to(DEVICE)
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"模型加载完成，设备：{DEVICE}")
    
    # 3. 加载测试集
    test_dataset = CommentDataset(
        data_path=TEST_DATA_PATH,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    print(f"测试集加载完成：共{len(test_dataset)}条样本")
    
    # 4. 模型评测
    test_metrics = compute_metrics(model, test_loader, DEVICE)
    
    # 5. 打印评测结果
    print("\n===== 测试集最终结果 =====")
    print(f"情感分类准确率：{test_metrics['emotion_acc']:.4f}")
    print(f"情感分类加权F1值：{test_metrics['emotion_f1']:.4f}")
    print(f"属性分类准确率：{test_metrics['attribute_acc']:.4f}")
    print(f"属性分类加权F1值：{test_metrics['attribute_f1']:.4f}")
    print(f"综合准确率：{(test_metrics['emotion_acc'] + test_metrics['attribute_acc'])/2:.4f}")
    
    # 6. 保存指标和可视化结果
    save_metrics(test_metrics, METRICS_SAVE_PATH)
    plot_confusion_matrix(
        true_labels=test_metrics["emotion_labels"],
        pred_labels=test_metrics["emotion_preds"],
        save_path=CONFUSION_MATRIX_PATH
    )
    print(f"\n指标结果已保存至：{METRICS_SAVE_PATH}")
    print(f"混淆矩阵已保存至：{CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    evaluate_model()