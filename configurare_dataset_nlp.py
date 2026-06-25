import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
import re

# ============================================================
# 1. CONFIGURARE CĂI
# ============================================================
ROOT_DIR = Path("D:/MIMIC")

# Fisierele din MIMIC-CXR-JPG (etichete, metadate, split)
CXR_JPG_DIR = ROOT_DIR / "MIMIC-CXR-JPG"
metadata_csv_path = CXR_JPG_DIR / "mimic-cxr-2.0.0-metadata.csv"
chexpert_csv_path = CXR_JPG_DIR / "mimic-cxr-2.0.0-chexpert.csv"
split_csv_path = CXR_JPG_DIR / "mimic-cxr-2.0.0-split.csv"

# Fisierele din Urgențe
ED_DIR = ROOT_DIR / "MIMIC-IV-ED"
edstays_csv_path = ED_DIR / "edstays.csv"

# Fisierele din MIMIC-CXR (Rapoarte Text)
CXR_REPORTS_DIR = ROOT_DIR / "MIMIC-CXR"
study_list_csv_path = CXR_REPORTS_DIR / "cxr-study-list.csv.gz"  # Fisierul sugerat de tine
reports_root = CXR_REPORTS_DIR  # Aici presupunem ca folderul 'files' e in 'MIMIC-CXR'

# ============================================================
# 2. VERIFICARE PRELIMINARĂ
# ============================================================
print(f"--- VERIFICARE CĂI (SISTEM: {os.name}) ---")
files_to_check = [
    ("METADATA", metadata_csv_path),
    ("CHEXPERT", chexpert_csv_path),
    ("SPLIT", split_csv_path),
    ("ED STAYS", edstays_csv_path),
    ("STUDY LIST", study_list_csv_path)
]

paths_ok = True
for name, p in files_to_check:
    if not p.exists():
        print(f"EROARE: NU GĂSESC {name} LA: {p}")
        paths_ok = False
    else:
        print(f"CONFIRMAT: GĂSIT {name}")

if not (reports_root / "files").exists():
    print(f"EROARE: NU GĂSESC FOLDERUL 'files' ÎN: {reports_root}")
    paths_ok = False

if not paths_ok:
    print("SISTEMUL SE OPREȘTE. VERIFICĂ CĂILE.")
    sys.exit()


# ============================================================
# 3. FUNCȚIE DE EXTRAGERE TEXT (NLP)
# ============================================================
def extract_report_text(report_path):
    """Extrage sectiunile FINDINGS sau IMPRESSION din textul brut si converteste in lowercase."""
    if not os.path.exists(report_path):
        return None

    with open(report_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Căutăm secțiunea FINDINGS
    findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', text, re.IGNORECASE | re.DOTALL)
    if findings_match and findings_match.group(1).strip():
        return findings_match.group(1).strip().lower()

    # Dacă nu găsim FINDINGS, încercăm IMPRESSION
    impression_match = re.search(r'IMPRESSION:(.*?)(?:$)', text, re.IGNORECASE | re.DOTALL)
    if impression_match and impression_match.group(1).strip():
        return impression_match.group(1).strip().lower()

    # Daca lipsesc ambele, luam tot textul curatat
    return text.strip().lower()


# ============================================================
# 4. ÎNCĂRCARE, SINCRONIZARE ED ȘI FILTRARE
# ============================================================
print("\n--- ÎNCEPE PROCESAREA DATELOR (ED ONLY) ---")

df_metadata = pd.read_csv(metadata_csv_path, usecols=['subject_id', 'study_id', 'StudyDate', 'StudyTime'])
df_edstays = pd.read_csv(edstays_csv_path, usecols=['subject_id', 'stay_id', 'intime', 'outtime'])

# Creare datetime pentru studii radiologice
date_str = df_metadata['StudyDate'].astype(float).astype(int).astype(str)
time_str = df_metadata['StudyTime'].astype(float).astype(int).astype(str).str.zfill(6)
df_metadata['study_datetime'] = pd.to_datetime(date_str + " " + time_str, format='%Y%m%d %H%M%S', errors='coerce')

df_edstays['intime'] = pd.to_datetime(df_edstays['intime'], errors='coerce')
df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'], errors='coerce')

# Fuziune Metadata + ED Stays și Filtrare
df_merged = pd.merge(df_metadata, df_edstays, on='subject_id', how='inner')
df_ed_studies = df_merged.query('study_datetime >= intime and study_datetime <= outtime').copy()
df_ed_studies = df_ed_studies.drop_duplicates(subset=['study_id'])
print(f"   Studii valide (în Urgențe): {len(df_ed_studies)}")

# ============================================================
# 5. PREPROCESARE ETICHETE (STRATEGIA U-ZEROS)
# ============================================================
label_cols = [
    'study_id', 'No Finding', 'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]

df_chexpert = pd.read_csv(chexpert_csv_path, usecols=label_cols)
pathology_cols = [c for c in label_cols if c != 'study_id']

# Strategia U-Zeros: NaN -> 0.0 și -1.0 -> 0.0
df_chexpert[pathology_cols] = df_chexpert[pathology_cols].fillna(0.0)
df_chexpert[pathology_cols] = df_chexpert[pathology_cols].replace(-1.0, 0.0)
df_chexpert = df_chexpert.drop_duplicates(subset=['study_id'])

# ============================================================
# 6. EXTRAGEREA CĂILOR CĂTRE RAPOARTE ȘI A TEXTULUI
# ============================================================
df_study_list = pd.read_csv(study_list_csv_path)  # Contine subject_id, study_id, path

# Fuzionăm toate informațiile
master_df = pd.merge(df_ed_studies, df_chexpert, on='study_id', how='inner')
master_df = pd.merge(master_df, df_study_list[['study_id', 'path']], on='study_id', how='inner')

df_split = pd.read_csv(split_csv_path, usecols=['study_id', 'split']).drop_duplicates(subset=['study_id'])
master_df = pd.merge(master_df, df_split, on='study_id', how='left')

print("\n--- EXTRAGERE TEXT DIN FIȘIERE ---")
valid_records = []

for index, row in master_df.iterrows():
    # Calea relativă vine din cxr-study-list, ex: "files/p10/p10000032/s50414267.txt"
    # Daca 'path' din CSV nu contine 'files/', adaugam manual. De regula, contine.
    rel_path = str(row['path'])
    full_report_path = reports_root / rel_path

    extracted_text = extract_report_text(full_report_path)

    if extracted_text:
        row_dict = row.to_dict()
        row_dict['report_text'] = extracted_text
        valid_records.append(row_dict)

final_df = pd.DataFrame(valid_records)

# ============================================================
# 7. SALVARE CSV MASTER
# ============================================================
output_filename = "nlp_ed_master_dataset.csv"
final_df.to_csv(output_filename, index=False)

print("\n" + "=" * 50)
print(f"SUCCES: FIȘIER SALVAT CA: {output_filename}")
print(f"TOTAL RAPOARTE PROCESATE: {len(final_df)}")
print("=" * 50)