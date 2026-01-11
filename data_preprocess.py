import pandas as pd
import jieba
import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE  # 解决数据不平衡问题
import os

# -------------------------- 1. 数据预处理主函数（核心适配逻辑）--------------------------
def preprocess_data(raw_data_path, processed_data_dir):
    """
    适配 social_ecommerce_data.csv 的预处理函数
    raw_data_path: 原始数据集路径（如 ./data/social_ecommerce_data.csv）
    processed_data_dir: 预处理后数据保存目录（如 ./data/processed_data）
    """
    # 1. 创建保存目录（若不存在）
    os.makedirs(processed_data_dir, exist_ok=True)
    print(f"开始处理数据集：{raw_data_path}")

    # 2. 加载原始数据（适配数据集字段名）
    df = pd.read_csv(raw_data_path)
    print(f"原始数据规模：{len(df)} 条")

    # 3. 基础数据清洗：处理空值、过滤无效评论
    # 3.1 填充辅助字段空值（不影响核心任务）
    df["comment_time"] = df["comment_time"].fillna("2024-01-01 00:00:00")  # 时间字段默认值
    df["product_brand"] = df["product_brand"].fillna("Unknown")  # 品牌字段默认值
    # 3.2 删除核心字段（评论内容）为空的行
    df = df.dropna(subset=["comment_text"])
    # 3.3 过滤特殊字符，仅保留中英文、数字（避免乱码影响分词）
    df["comment_text"] = df["comment_text"].apply(
        lambda x: re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", str(x)).strip()
    )
    # 3.4 删除空字符串评论（清洗后可能产生）
    df = df[df["comment_text"].str.len() > 2]  # 保留长度≥3的有效评论
    print(f"清洗后数据规模：{len(df)} 条")

    # 4. 中文分词与停用词过滤（提升模型效率）
    # 4.1 加载停用词（需确保 ./data/stopwords.txt 存在，无则可注释此部分）
    try:
        stopwords = pd.read_csv("./data/stopwords.txt", names=["word"], encoding="utf-8")["word"].tolist()
    except FileNotFoundError:
        print("未找到 stopwords.txt，跳过停用词过滤")
        stopwords = []
    # 4.2 分词（保留长度≥2的词，过滤停用词）
    df["分词结果"] = df["comment_text"].apply(
        lambda x: " ".join([w for w in jieba.lcut(x) if w not in stopwords and len(w) >= 2])
    )
    # 4.3 删除分词后为空的行（极端情况，如评论全是停用词）
    df = df[df["分词结果"].str.len() > 0]
    print(f"分词后有效数据规模：{len(df)} 条")

    # 5. 标签编码（核心适配：将数据集字段映射为项目所需标签）
    # 5.1 情感标签：1-5分 → 0-4（非常不满-非常满意）
    def map_emotion(rating):
        if rating == 5:
            return 4  # 非常满意
        elif rating == 4:
            return 3  # 满意
        elif rating == 3:
            return 2  # 中性
        elif rating == 2:
            return 1  # 不满
        elif rating == 1:
            return 0  # 非常不满
        else:
            return 2  # 异常评分默认归为中性

    df["情感标签"] = df["user_rating"].apply(map_emotion)

    # 5.2 商品属性标签：product_category → 0-4（品类映射）
    category_mapping = {
        "clothing": 0,      # 服装类
        "electronics": 1,   # 电子产品类
        "food": 2,          # 食品类
        "home_goods": 3,    # 家居用品类
        "Unknown": 4        # 未知品类（避免映射失败）
    }
    # 先将空品类填充为"Unknown"，再映射
    df["product_category"] = df["product_category"].fillna("Unknown")
    df["属性标签"] = df["product_category"].map(category_mapping)
    # 确保标签为整数类型
    df["情感标签"] = df["情感标签"].astype(int)
    df["属性标签"] = df["属性标签"].astype(int)

    # 6. 处理训练集数据不平衡（SMOTE过采样，提升小众情感类别的模型效果）
    print("开始处理数据不平衡问题（SMOTE过采样）")
    # 6.1 提取训练集特征（用分词长度作为临时特征，适配SMOTE输入格式）
    train_temp, _ = train_test_split(df, test_size=0.3, random_state=42, stratify=df["情感标签"])
    X_train = train_temp["分词结果"].apply(lambda x: len(x.split())).values.reshape(-1, 1)
    y_train = train_temp["情感标签"].values

    # 6.2 SMOTE过采样（仅对训练集操作，避免数据泄露）
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # 6.3 构建过采样后的训练集（保持数据分布一致性）
    train_smote = train_temp.sample(n=len(X_train_smote), replace=True, random_state=42)
    train_smote["情感标签"] = y_train_smote

    # 7. 数据集划分（训练集:验证集:测试集 = 7:1:2）
    # 7.1 先划分训练集（含过采样）和临时集（验证+测试）
    temp_df = df.drop(train_temp.index)  # 临时集 = 原数据 - 初始训练集
    # 7.2 划分验证集和测试集（从临时集中分）
    val_df, test_df = train_test_split(
        temp_df, test_size=0.6667, random_state=42, stratify=temp_df["情感标签"]
    )

    # 8. 保存预处理后的数据（供后续训练/评测使用）
    train_smote.to_csv(f"{processed_data_dir}/train.csv", index=False, encoding="utf-8")
    val_df.to_csv(f"{processed_data_dir}/val.csv", index=False, encoding="utf-8")
    test_df.to_csv(f"{processed_data_dir}/test.csv", index=False, encoding="utf-8")

    # 9. 输出预处理结果日志（便于核对）
    print("\n=== 预处理完成 ===")
    print(f"训练集（过采样后）：{len(train_smote)} 条 | 情感标签分布：{train_smote['情感标签'].value_counts().sort_index().to_dict()}")
    print(f"验证集：{len(val_df)} 条 | 情感标签分布：{val_df['情感标签'].value_counts().sort_index().to_dict()}")
    print(f"测试集：{len(test_df)} 条 | 情感标签分布：{test_df['情感标签'].value_counts().sort_index().to_dict()}")
    print(f"数据保存路径：{processed_data_dir}")


# -------------------------- 2. 自定义数据集类（供PyTorch加载）--------------------------
class CommentDataset(Dataset):
    """
    适配预处理后数据的数据集类，用于PyTorch的DataLoader加载
    """
    def __init__(self, data_path, tokenizer, max_len=128):
        self.df = pd.read_csv(data_path, encoding="utf-8")
        self.tokenizer = tokenizer  # BERT分词器（来自transformers库）
        self.max_len = max_len      # 文本最大长度（超过截断，不足填充）

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.df)

    def __getitem__(self, idx):
        """按索引获取单个样本（返回模型可接受的张量格式）"""
        # 1. 提取当前样本的核心数据
        text = self.df.iloc[idx]["分词结果"]  # 已分词的文本
        emotion_label = self.df.iloc[idx]["情感标签"]  # 情感标签（0-4）
        attribute_label = self.df.iloc[idx]["属性标签"]  # 属性标签（0-4）

        # 2. BERT分词器编码（转换为input_ids和attention_mask）
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # 添加[CLS]和[SEP]特殊标记
            max_length=self.max_len,
            padding="max_length",     # 填充到最大长度
            truncation=True,          # 截断超过最大长度的文本
            return_tensors="pt"       # 返回PyTorch张量
        )

        # 3. 整理返回格式（展平张量，避免维度冗余）
        return {
            "input_ids": encoding["input_ids"].flatten(),  # shape: (max_len,)
            "attention_mask": encoding["attention_mask"].flatten(),  # shape: (max_len,)
            "emotion_labels": torch.tensor(emotion_label, dtype=torch.long),  # 情感标签张量
            "attribute_labels": torch.tensor(attribute_label, dtype=torch.long)  # 属性标签张量
        }


# -------------------------- 3. 主函数（直接运行脚本时执行预处理）--------------------------
if __name__ == "__main__":
    # 配置路径（需根据你的项目实际结构调整，默认适配原项目目录）
    RAW_DATA_PATH = "./data/social_ecommerce_data.csv"  # 原始数据集路径
    PROCESSED_DIR = "./data/processed_data"             # 预处理后数据保存目录

    # 执行预处理
    preprocess_data(
        raw_data_path=RAW_DATA_PATH,
        processed_data_dir=PROCESSED_DIR
    )