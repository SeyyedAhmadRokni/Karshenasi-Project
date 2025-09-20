import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import time
from tqdm import tqdm

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

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

def load_and_preprocess_data(frac=0.1):  # 10% از کل دیتا برای تست
    log(f"Loading {frac*100}% of full dataset...")
    
    # بارگذاری داده‌ها
    data_path = '../'
    vle = pd.read_csv(data_path + 'vle.csv')
    student_vle = pd.read_csv(data_path + 'studentVle.csv')
    student_info = pd.read_csv(data_path + 'studentInfo.csv')
    
    # نمونه‌گیری برای تست
    student_vle_sample = student_vle.sample(frac=frac, random_state=123)  # seed متفاوت از ترین
    
    # ادغام
    data = student_vle_sample.merge(vle, on=['id_site', 'code_module', 'code_presentation'], how='inner')
    data = data.merge(student_info, on=['id_student', 'code_module', 'code_presentation'], how='inner')
    
    log(f"Test dataset size: {data.shape}")
    
    # پردازش سریع
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())
    
    categorical_cols = ['code_module', 'code_presentation', 'gender', 'region',
                        'highest_education', 'imd_band', 'age_band',
                        'disability', 'final_result']
    
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype('category').cat.codes
    
    # ایجاد متن
    data['text_description'] = (
        "Student " + data['id_student'].astype(str) +
        " in course " + data['code_module'].astype(str) +
        " clicked " + data['sum_click'].astype(str) +
        " times on " + data['activity_type'].astype(str) + "."
    )
    
    return data

def test_model(model_path='./bert_classification_model_final', test_frac=0.1):
    log("Starting model test...")
    
    # بارگذاری مدل
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # آماده‌سازی دیتا
    test_data = load_and_preprocess_data(frac=test_frac)
    texts = test_data['text_description'].values
    labels = test_data['final_result'].values
    
    log(f"Testing on {len(texts)} samples")
    
    # ایجاد DataLoader
    dataset = TextDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # تست
    all_preds = []
    all_labels = []
    confidences = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Testing"), start=1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            confidences.extend(torch.max(probs, dim=1)[0].cpu().numpy())

            # ✅ Log every 200 batches
            if i % 200 == 0:
                acc_so_far = accuracy_score(all_labels, all_preds)
                f1_so_far = f1_score(all_labels, all_preds, average='macro')
                avg_conf_so_far = np.mean(confidences)
                log(f"[Batch {i}] Accuracy: {acc_so_far:.4f}, "
                    f"F1 (macro): {f1_so_far:.4f}, "
                    f"Avg Confidence: {avg_conf_so_far:.4f}")

    # نتایج
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    avg_confidence = np.mean(confidences)
    
    print("\n" + "="*40)
    print("TEST RESULTS")
    print("="*40)
    print(f"Test Samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print("="*40)
    
    # مقایسه با ترین (اگر موجود باشه)
    try:
        train_df = pd.read_csv('train.csv')
        log("Comparing with training data...")
        
        train_texts = train_df['text_description'].values[:1000]  # فقط 1000 تا برای سرعت
        train_labels = train_df['final_result'].values[:1000]
        
        train_dataset = TextDataset(train_texts, train_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        
        train_preds = []
        train_true = []
        
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Testing on train data"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(labels_batch.cpu().numpy())
        
        train_acc = accuracy_score(train_true, train_preds)
        
        print(f"\nTraining Accuracy (sample): {train_acc:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Difference: {train_acc - accuracy:.4f}")
        
        if train_acc - accuracy > 0.1:
            print("⚠️  WARNING: Possible overfitting!")
        else:
            print("✅ Good generalization")
            
    except FileNotFoundError:
        log("train.csv not found, skipping comparison")
    
    # ذخیره نتایج
    results_df = pd.DataFrame({
        'text': texts,
        'true_label': all_labels,
        'predicted_label': all_preds,
        'confidence': confidences
    })
    results_df.to_csv('test_results.csv', index=False)
    log("Results saved to test_results.csv")
    
    return accuracy, f1, avg_confidence

if __name__ == "__main__":
    # تست مدل روی 10% از کل دیتا (حدود 500K نمونه)
    test_model(test_frac=0.05)