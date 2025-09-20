import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW
import warnings
import os
import time
from tqdm import tqdm

# تنظیمات محیطی برای دیباگ بهتر CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # برای تشخیص synchronous errors در CUDA

warnings.filterwarnings('ignore')

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# کلاس دیتاست سفارشی برای BERT
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# تابع برای استخراج امبدینگ‌ها از BERT تیون‌شده
def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model.bert(input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(cls_embeddings.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    return np.vstack(embeddings), np.array(labels)

# تابع برای ارزیابی مدل
def evaluate_model(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    log(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")


# روش اول: BERT مخصوص classification (BertForSequenceClassification)
def method_1_bert_classification(train_df_path='train.csv', test_df_path='test.csv', start_epoch=0, num_epochs=3,
                                 batch_size=32, learning_rate=2e-5):
    log("Starting Method 1: BERT for Sequence Classification")

    # لود داده‌های train و test ذخیره‌شده
    train_df = pd.read_csv(train_df_path)
    test_df = pd.read_csv(test_df_path)

    train_texts = train_df['text_description'].values
    train_labels = train_df['final_result'].values
    test_texts = test_df['text_description'].values
    test_labels = test_df['final_result'].values

    num_labels = len(np.unique(train_labels))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # لود مدل اگر start_epoch > 0
    if start_epoch > 0:
        load_path = f'./bert_classification_model_epoch_{start_epoch}'
        model = BertForSequenceClassification.from_pretrained(load_path, num_labels=num_labels)
        optimizer_state = torch.load(f'{load_path}/optimizer.pt')
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(optimizer_state)
        log(f"Loaded model and optimizer from {load_path}")
    else:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        optimizer = AdamW(model.parameters(), lr=learning_rate)

    model.to(device)

    # آموزش مدل از start_epoch
    model.train()
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        batch_idx = 0
        running_loss = 0.0
        running_preds = []
        running_labels = []
        for batch in tqdm(train_loader):
            batch_idx += 1
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            running_preds.extend(preds.cpu().numpy())
            running_labels.extend(labels.cpu().numpy())
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                avg_loss = running_loss / 50
                acc = accuracy_score(running_labels, running_preds)
                prec = precision_score(running_labels, running_preds, average='macro')
                rec = recall_score(running_labels, running_preds, average='macro')
                f1 = f1_score(running_labels, running_preds, average='macro')
                log(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}, Avg Loss: {avg_loss:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
                running_loss = 0.0  # ریست running_loss بعد از چاپ
                running_preds = []  # ریست لیست پیش‌بینی‌ها
                running_labels = []  # ریست لیست لیبل‌ها

        log(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss / len(train_loader):.4f}")

        # سیو مدل بعد از هر epoch روی لوکال
        save_path = f'./bert_classification_model_epoch_{epoch + 1}'
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        torch.save(optimizer.state_dict(), f'{save_path}/optimizer.pt')
        log(f"Model and optimizer saved to {save_path} after epoch {epoch + 1}")

    # ارزیابی
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        evaluate_model(true_labels, preds)

        # ذخیره نهایی مدل روی لوکال
        final_save_path = './bert_classification_model_final'
        model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        torch.save(optimizer.state_dict(), f'{final_save_path}/optimizer.pt')
        log("Method 1 completed and final model saved locally.")

if __name__ == "__main__":
    # برای ادامه آموزش، start_epoch رو تغییر بده، مثلاً اگر از epoch 2 ادامه می‌دی، start_epoch=1 (چون بعد از epoch 1 سیو شده)
    # روش اول
    method_1_bert_classification(start_epoch=0)
