import torch
import pandas as pd
from tqdm import tqdm
from kobert_transformers import get_tokenizer
from torch.nn.functional import softmax
from torch import nn


# KoBERT 분류 모델 정의
class KoBERTClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super(KoBERTClassifier, self).__init__()
        from kobert_transformers import get_kobert_model
        self.bert = get_kobert_model()
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # [CLS] 토큰 사용
        return logits


# 분석 함수 정의
def analyze_crawled_news():
    # 학습된 모델 로드
    model = KoBERTClassifier(num_labels=3)
    model.load_state_dict(torch.load("sentiment_model_using_review.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    tokenizer = get_tokenizer()

    # 데이터 로드
    df = pd.read_csv("../crawl/crawling_summary.csv")
    texts = (df["title"] + " " + df["summary"]).tolist()

    # 결과 저장
    labels = []
    for text in tqdm(texts, desc="Analyzing"):
        encoding = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask)
            probs = softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            labels.append(pred)

    # 레이블 추가 및 저장
    df["label"] = labels
    df.to_csv("crawled_summary_with_label.csv", index=False, encoding="utf-8-sig")
    print("Saved results to crawled_news_with_label.csv")


# MPS를 위한 DEVICE 설정
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MAX_LEN = 128  # 최대 토큰 길이


if __name__ == "__main__":
    analyze_crawled_news()