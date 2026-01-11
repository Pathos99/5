import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import os
from model import BertSentimentAttributeModel
from data_preprocess import CommentDataset
from utils import compute_metrics

# 训练参数配置（可根据设备性能调整）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择CPU/GPU
BATCH_SIZE = 16 if DEVICE.type == "cpu" else 32  # CPU用16，GPU用32
EPOCHS = 15  # 训练轮数
LEARNING_RATE = 1e-5  # 学习率
MAX_LEN = 128  # 文本最大长度
MODEL_SAVE_PATH = "./results/model_weights/best_model.pth"  # 最优模型保存路径

def train_model():
    # 1. 初始化BERT分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    print(f"分词器初始化完成，设备：{DEVICE}")
    
    # 2. 加载数据集
    train_dataset = CommentDataset(
        data_path="./data/processed_data/train.csv",
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_dataset = CommentDataset(
        data_path="./data/processed_data/val.csv",
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # 3. 创建数据加载器（批量加载数据，支持多线程）
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 训练集打乱
        num_workers=2  # 多线程加载（CPU推荐2，GPU可设4）
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 验证集不打乱
        num_workers=2
    )
    print(f"数据加载完成：训练集{len(train_dataset)}条，验证集{len(val_dataset)}条")
    
    # 4. 初始化模型并移到指定设备
    model = BertSentimentAttributeModel().to(DEVICE)
    
    # 5. 优化器和学习率调度器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4  # 权重衰减，正则化
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10, eta_min=1e-6  # 余弦退火调度
    )
    
    # 6. 训练循环（早停策略：连续3轮验证集准确率无提升则停止）
    best_val_acc = 0.0
    early_stop_count = 0
    
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()  # 模型切换到训练模式
        train_total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # 提取batch数据并移到设备
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            emotion_labels = batch["emotion_labels"].to(DEVICE)
            attribute_labels = batch["attribute_labels"].to(DEVICE)
            
            # 梯度清零
            optimizer.zero_grad()
            # 模型前向传播
            emotion_logits, attribute_logits, loss = model(
                input_ids, attention_mask, emotion_labels, attribute_labels
            )
            # 反向传播计算梯度
            loss.backward()
            # 优化器更新参数
            optimizer.step()
            
            # 累计训练损失
            train_total_loss += loss.item()
            
            # 打印训练进度（每100个batch打印一次）
            if (batch_idx + 1) % 100 == 0:
                avg_loss = train_total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1} | 平均训练损失：{avg_loss:.4f}")
        
        # 验证阶段
        model.eval()  # 模型切换到评估模式
        val_metrics = compute_metrics(model, val_loader, DEVICE)
        val_avg_acc = (val_metrics["emotion_acc"] + val_metrics["attribute_acc"]) / 2  # 综合准确率
        
        # 打印验证结果
        print(f"\nEpoch {epoch+1} 验证结果：")
        print(f"情感分类准确率：{val_metrics['emotion_acc']:.4f} | F1值：{val_metrics['emotion_f1']:.4f}")
        print(f"属性分类准确率：{val_metrics['attribute_acc']:.4f} | F1值：{val_metrics['attribute_f1']:.4f}")
        print(f"综合准确率：{val_avg_acc:.4f}\n")
        
        # 保存最优模型
        if val_avg_acc > best_val_acc:
            best_val_acc = val_avg_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)  # 保存模型权重
            print(f"最优模型已更新，当前最佳综合准确率：{best_val_acc:.4f}")
            early_stop_count = 0  # 重置早停计数器
        else:
            early_stop_count += 1
            print(f"早停计数器：{early_stop_count}/3")
            if early_stop_count >= 3:
                print("连续3轮验证集准确率无提升，触发早停！")
                break
        
        # 学习率调度器更新
        scheduler.step()
    
    print("训练完成！最优模型已保存至：", MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()