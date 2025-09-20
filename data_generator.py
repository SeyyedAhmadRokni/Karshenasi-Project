import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import os
import time

# تنظیمات محیطی برای دیباگ بهتر CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # برای تشخیص synchronous errors در CUDA

warnings.filterwarnings('ignore')


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# 1. بارگذاری و ادغام داده‌ها
def load_and_merge_data(frac=0.5):
    log("Loading CSV files...")
    data_path = '../'
    vle = pd.read_csv(data_path + 'vle.csv')
    student_vle = pd.read_csv(data_path + 'studentVle.csv')
    student_info = pd.read_csv(data_path + 'studentInfo.csv')

    if frac < 1.0:
        student_vle_sample = student_vle.sample(frac=frac, random_state=42)
        log(f"Sampled {len(student_vle_sample)} rows from studentVle")
    else:
        student_vle_sample = student_vle

    data = student_vle_sample.merge(vle, on=['id_site', 'code_module', 'code_presentation'], how='inner')
    data = data.merge(student_info, on=['id_student', 'code_module', 'code_presentation'], how='inner')
    log(f"Merged dataset size: {data.shape}")
    return data


# 2. پیش‌پردازش داده‌ها
def preprocess_data(data):
    log("Preprocessing data...")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    object_cols = data.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        data[col] = data[col].fillna(data[col].median())

    for col in object_cols:
        mode_val = data[col].mode()[0] if not data[col].mode().empty else np.nan
        data[col] = data[col].fillna(mode_val)

    categorical_cols = ['code_module', 'code_presentation', 'gender', 'region',
                        'highest_education', 'imd_band', 'age_band',
                        'disability', 'final_result']

    for col in categorical_cols:
        data[col] = data[col].astype('category').cat.codes

    fslsm_mapping = {
        'forumng': 'Active', 'oucollaborate': 'Active', 'ouwiki': 'Active', 'glossary': 'Active',
        'htmlactivity': 'Active',
        'oucontent': 'Sensing', 'questionnaire': 'Sensing', 'quiz': 'Sensing', 'externalquiz': 'Sensing',
        'dataplus': 'Visual', 'dualpane': 'Visual', 'folder': 'Visual', 'page': 'Visual', 'homepage': 'Visual',
        'resource': 'Visual', 'url': 'Visual', 'ouelluminate': 'Visual', 'subpage': 'Visual',
        'repeatactivity': 'Sequential', 'sharedsubpage': 'Global'
    }
    data['learning_style'] = data['activity_type'].map(fslsm_mapping).fillna('Unknown')
    data = data[data['learning_style'] != 'Unknown']

    data['text_description'] = (
            "Student " + data['id_student'].astype(str) +
            " in course " + data['code_module'].astype(str) +
            " clicked " + data['sum_click'].astype(str) +
            " times on " + data['activity_type'].astype(str) + "."
    )
    log("Data preprocessing complete.")
    return data


if __name__ == "__main__":
    data = load_and_merge_data(frac=0.02)  # frac کوچک برای تست سریع، می‌تونی تغییر بدی
    processed_data = preprocess_data(data)

    # اسپلیت داده‌ها به train و test و ذخیره آن‌ها
    texts = processed_data['text_description'].values
    labels = processed_data['final_result'].values
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    train_df = pd.DataFrame({'text_description': train_texts, 'final_result': train_labels})
    test_df = pd.DataFrame({'text_description': test_texts, 'final_result': test_labels})

    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    log("Train and test data saved to train.csv and test.csv")
