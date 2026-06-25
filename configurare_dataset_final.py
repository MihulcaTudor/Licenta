import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
import re

# 1 CONFIGURARE CAI
ROOT_DIR = Path("D:/MIMIC")
CXR_JPG_DIR = ROOT_DIR / "MIMIC-CXR-JPG"
CXR_REPORTS_DIR = ROOT_DIR / "MIMIC-CXR"

data_root = CXR_JPG_DIR / "mimic-cxr-jpg/2.1.0/files"
metadata_csv_path = CXR_JPG_DIR / "mimic-cxr-2.0.0-metadata.csv"
chexpert_csv_path = CXR_JPG_DIR / "mimic-cxr-2.0.0-chexpert.csv"
study_list_csv_path = CXR_REPORTS_DIR / "cxr-study-list.csv.gz"

# 2 FUNCTIE DE EXTRAGERE TEXT (NLP)
def extract_report_text(report_path):
    """Extrage secțiunile FINDINGS sau IMPRESSION din textul brut"""
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, 'r', encoding='utf-8') as file:
            text = file.read()
        #cautam seciunea findings
        findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', text, re.IGNORECASE | re.DOTALL)
        if findings_match and findings_match.group(1).strip():
            return findings_match.group(1).strip().lower()
        #daca nu gasim findings, incercam impresion
        impression_match = re.search(r'IMPRESSION:(.*?)(?:$)', text, re.IGNORECASE | re.DOTALL)
        if impression_match and impression_match.group(1).strip():
            return impression_match.group(1).strip().lower()
        #daca nu gasim niciuna, luam tot textul curatat
        return text.strip().lower()
    except Exception:
        return None

# 3 VERIFICARE
# print(f"--- VERIFICARE CAI (SISTEM: {os.name}) ---")
files_to_check = [
    ("METADATA", metadata_csv_path),
    ("CHEXPERT", chexpert_csv_path),
    ("STUDY LIST", study_list_csv_path)
]

paths_ok = True
for name, p in files_to_check:
    if not p.exists():
        print(f"EROARE: NU GASESC {name} LA: {p}")
        paths_ok = False
    else:
        print(f"CONFIRMAT: GASIT {name}")

if not data_root.exists():
    print(f"EROARE: FOLDERUL DE IMAGINI NU ESTE LA: {data_root}")
    paths_ok = False
else:
    print("CONFIRMAT: GASIT FOLDER IMAGINI")

if not paths_ok:
    print("SISTEMUL SE OPRESTE. VERIFICA CAILE.")
    sys.exit()

# 4 SCANARE IMAGINI DE PE DISC
print(f"\n--- SCANARE FOLDER IMAGINI: {data_root} ---")
all_files_data = []
for path in data_root.rglob('*.jpg'):
    all_files_data.append({'image_path': str(path), 'dicom_id': path.stem})

df_images = pd.DataFrame(all_files_data)
# print(f"Total fisiere JPG gasite fizic: {len(df_images)}")

# 5 INCARCARE METADATE SI ASOCIERE
print("---- ASOCIERE METADATE ---")
df_metadata = pd.read_csv(metadata_csv_path, usecols=['dicom_id', 'study_id', 'subject_id'])
master_df = pd.merge(df_images, df_metadata, on='dicom_id', how='inner')

# 6: PROCESARE ETICHETE CHEXPERT
print("---- APLICARE  ETICHETE ---")
label_cols = [
    'study_id', 'No Finding', 'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]

df_chexpert = pd.read_csv(chexpert_csv_path, usecols=label_cols)
pathology_cols = [c for c in label_cols if c != 'study_id']

df_chexpert[pathology_cols] = df_chexpert[pathology_cols].fillna(0.0)
df_chexpert[pathology_cols] = df_chexpert[pathology_cols].replace(-1.0, np.nan)
df_chexpert = df_chexpert.drop_duplicates(subset=['study_id'])

master_df = pd.merge(master_df, df_chexpert, on='study_id', how='inner')

#  7: ADAUGARE CAI CATRE RAPOARTE
print("----- SINCRONIZARE CU RAPOARTELE TEXT -----")
df_reports = pd.read_csv(study_list_csv_path, usecols=['study_id', 'path'])
df_reports['report_path'] = df_reports['path'].apply(lambda x: str(CXR_REPORTS_DIR / x))
df_reports = df_reports.drop(columns=['path'])

master_df = pd.merge(master_df, df_reports, on='study_id', how='inner')

# 8: EXTRAGEREA TEXTULUI RELEVANT DIN RAPOARTE
print("--- EXTRAGERE TEXT DIN RAPOARTE  ---")
master_df['report_text'] = master_df['report_path'].apply(extract_report_text)

master_df = master_df.dropna(subset=['report_text'])
print(f"Rapoarte procesate cu succes: {len(master_df)}")

# 9: CUSTOM SPLIT PE PACIENT
print("--- GENERARE SPLIT 70/15/15 (TRAIN/VAL/TEST) ---")
unique_subjects = master_df['subject_id'].unique()

np.random.seed(42)
np.random.shuffle(unique_subjects)

train_idx = int(len(unique_subjects) * 0.70)
val_idx = int(len(unique_subjects) * 0.85)

train_subjects = unique_subjects[:train_idx]
val_subjects = unique_subjects[train_idx:val_idx]
test_subjects = unique_subjects[val_idx:]

split_mapping = {}
split_mapping.update({subj: 'train' for subj in train_subjects})
split_mapping.update({subj: 'val' for subj in val_subjects})
split_mapping.update({subj: 'test' for subj in test_subjects})

master_df['split'] = master_df['subject_id'].map(split_mapping)

# 10: REORGANIZARE SI SALVARE
final_columns = ['subject_id', 'study_id', 'dicom_id', 'image_path', 'report_path', 'report_text', 'split'] + pathology_cols
master_df = master_df[final_columns]

print("\n" + "=" * 60)
print("STATISTICI FINALE DATASET")
print("=" * 60)
print(f"Total imagini/randuri (cu text extras): {len(master_df)}")
print(f"Total pacienti unici:     {master_df['subject_id'].nunique()}")
print(f"\nDistributie Split (Imagini):")
print(master_df['split'].value_counts())
print("=" * 60)

output_filename = "mimic_complete_master_dataset.csv"
master_df.to_csv(output_filename, index=False)

