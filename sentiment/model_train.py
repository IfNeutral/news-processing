import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from kobert_transformers import get_tokenizer, get_kobert_model
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn

# Device 설정 (MPS 지원)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
MAX_LEN = 75              # 문장의 최대 길이
BATCH_SIZE = 64            # 배치 크기
LEARNING_RATE = 5e-5       # 학습률
WARMUP_RATIO = 0.1         # 워밍업 비율
EPOCHS = 7                 # 에포크 수
MAX_GRAD_NORM = 1          # 그래디언트 클리핑


class KoBERTClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super(KoBERTClassifier, self).__init__()
        self.bert = get_kobert_model()  # KoBERT 모델 로드
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # 분류 헤드 추가

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return (loss, logits) if loss is not None else logits


# 데이터셋 정의
class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# 데이터 로드 및 전처리
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna()  # 결측치 제거
    texts = df["RawText"].tolist()
    labels = df["GeneralPolarity"].tolist()
    return texts, labels


# 학습
def train(model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()

        # 그래디언트 클리핑
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % 200 == 0:
            print(f"Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)


# 평가
def evaluate(model, test_loader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs  # 모델 출력

            # 출력 차원 확인 및 수정
            if logits.ndim == 1:
                logits = logits.unsqueeze(1)

            # 각 샘플의 예측 값 계산
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # 정확도 계산
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


def main():
    # 1. KoBERT 모델과 토크나이저 로드
    model = KoBERTClassifier(num_labels=3)  # 분류(네이버 영화 리뷰에는 긍정, 부정만 존재하기 때문에 추후 중립 의견 데이터셋을 찾으면 3으로 변경해야 함)
    tokenizer = get_tokenizer()
    model.to(DEVICE)

    # 2. 데이터 로드
    train_texts, train_labels = load_data("review_data/train_data.csv")
    test_texts, test_labels = load_data("review_data/evaluate_data.csv")

    # 3. Dataset 및 DataLoader 준비
    train_dataset = BERTDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    test_dataset = BERTDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 4. 옵티마이저와 스케줄러 설정
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 5. 학습
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        avg_loss = train(model, train_loader, optimizer, scheduler)
        print(f"Average Loss: {avg_loss:.4f}")

    # 6. 평가
    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")

    # 7. 모델 저장
    torch.save(model.state_dict(), "sentiment_model_using_review.pth")
    print("Model saved to sentiment_model_using_review.pth")


if __name__ == "__main__":
    main()