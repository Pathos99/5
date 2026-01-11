import torch
import torch.nn as nn
from transformers import BertModel

# 基于BERT的情感+属性分类模型
class BertSentimentAttributeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练BERT模型（中文版本）
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        # Dropout层（正则化，防止过拟合）
        self.dropout = nn.Dropout(0.15)  # 调整dropout率至0.15，增强泛化能力
        # 情感分类头（768是BERT输出维度，5是情感类别数）
        self.emotion_head = nn.Sequential(
            nn.Linear(768, 256),  # 全连接层：768→256
            nn.ReLU(),  # 激活函数
            nn.Linear(256, 5)  # 输出层：256→5
        )
        # 属性分类头（5是属性类别数）
        self.attribute_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
        # 损失函数（交叉熵损失，适用于分类任务）
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, emotion_labels=None, attribute_labels=None):
        # BERT模型前向传播：输入input_ids和attention_mask
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 提取[CLS] token的嵌入（用于分类任务）
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [batch_size, 768]
        # 经过Dropout层
        cls_embedding = self.dropout(cls_embedding)
        
        # 分别通过两个分类头，得到预测日志its
        emotion_logits = self.emotion_head(cls_embedding)
        attribute_logits = self.attribute_head(cls_embedding)
        
        # 计算损失（训练时计算，推理时不计算）
        loss = 0.0
        if emotion_labels is not None:
            loss += self.loss_fn(emotion_logits, emotion_labels)
        if attribute_labels is not None:
            loss += self.loss_fn(attribute_logits, attribute_labels)
        
        # 返回预测结果和损失（推理时损失为0）
        return emotion_logits, attribute_logits, loss