import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np

# ============================================================
# 1. CONFIGURARE CAI (WINDOWS)
# ============================================================
ROOT_DIR = Path("D:/MIMIC")
CXR_DIR = ROOT_DIR / "MIMIC-CXR-JPG"
ED_DIR = ROOT_DIR / "MIMIC-IV-ED"

# Calea catre imaginile JPG
data_root = CXR_DIR / "mimic-cxr-jpg/2.1.0/files"

# Fisierele CSV
split_csv_path = CXR_DIR / "mimic-cxr-2.0.0-split.csv"
metadata_csv_path = CXR_DIR / "mimic-cxr-2.0.0-metadata.csv"
chexpert_csv_path = CXR_DIR / "mimic-cxr-2.0.0-chexpert.csv"
edstays_csv_path = ED_DIR / "edstays.csv"

# ============================================================
# 2. VERIFICARE PRELIMINARA
# ============================================================
print(f"--- VERIFICARE CAI (SISTEM: {os.name}) ---")
paths_ok = True

files_to_check = [
    ("SPLIT CSV", split_csv_path),
    ("METADATA", metadata_csv_path),
    ("CHEXPERT", chexpert_csv_path),
    ("ED STAYS", edstays_csv_path)
]

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

# ============================================================
# 3. INCARCARE DATE
# ============================================================
print("\n--- INCEPE PROCESAREA DATELOR ---")

df_metadata = pd.read_csv(metadata_csv_path, usecols=['dicom_id', 'subject_id', 'study_id', 'StudyDate', 'StudyTime'])
df_edstays = pd.read_csv(edstays_csv_path, usecols=['subject_id', 'stay_id', 'intime', 'outtime'])
df_split = pd.read_csv(split_csv_path, usecols=['dicom_id', 'study_id', 'split'])

# Citim CheXpert Labels (Inclusiv Support Devices daca vrei)
label_cols = [
    'study_id', 'No Finding', 'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]

# Verificam ce coloane exista fizic in CSV
available_cols = list(pd.read_csv(chexpert_csv_path, nrows=1).columns)
use_cols = [c for c in label_cols if c in available_cols]
df_chexpert = pd.read_csv(chexpert_csv_path, usecols=use_cols)

# ============================================================
# 4. CURATARE SI FILTRARE (LOGICA CERUTA DE TINE)
# ============================================================
pathology_cols = [c for c in use_cols if c != 'study_id']

print("--- APLICARE LOGICA SPECIALE PENTRU ETICHETE ---")

# Pasul 1: Umplem NaN-urile ORIGINALE cu 0 (Lipsa mentiunii = Negativ)
# Acesta este standardul CheXpert: daca nu scrie nimic, e sanatos (0).
df_chexpert[pathology_cols] = df_chexpert[pathology_cols].fillna(0)

# Pasul 2: MODIFICARE CHEIE -> -1 devine NaN (CASUTA GOALA)
# Nu stergem randul, doar golim celula unde scrie -1.
df_chexpert[pathology_cols] = df_chexpert[pathology_cols].replace(-1, np.nan)

print("   -> Valorile de -1 au fost sterse (inlocuite cu NaN).")
print("   -> Randurile au fost pastrate.")

# Eliminam duplicatele pe baza de study_id
df_chexpert = df_chexpert.drop_duplicates(subset=['study_id'])

# ============================================================
# 5. PROCESARE TIMP SI MERGE
# ============================================================
print("--- PROCESARE TIMP SI FUZIUNE ---")

# Metadata datetime
date_str = df_metadata['StudyDate'].astype(float).astype(int).astype(str)
time_str = df_metadata['StudyTime'].astype(float).astype(int).astype(str).str.zfill(6)
df_metadata['study_datetime'] = pd.to_datetime(date_str + " " + time_str, format='%Y%m%d %H%M%S', errors='coerce')

# ED datetime
df_edstays['intime'] = pd.to_datetime(df_edstays['intime'], errors='coerce')
df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'], errors='coerce')

# Fuziune Metadata + ED
df_merged = pd.merge(df_metadata, df_edstays, on='subject_id', how='inner')

# Filtrare: Imaginea trebuie sa fie facuta in timpul vizitei la Urgenta (ED)
df_ed_images = df_merged.query('study_datetime >= intime and study_datetime <= outtime').copy()
df_ed_images = df_ed_images.drop_duplicates(subset=['dicom_id'])

print(f"   Imagini validate temporal (din ED): {len(df_ed_images)}")

# ============================================================
# 6. SCANARE DISC SI ASAMBLARE FINALA
# ============================================================
print(f"\n--- SCANARE FOLDER IMAGINI: {data_root} ---")
all_files_data = []
for path in data_root.rglob('*.jpg'):
    all_files_data.append({'image_path': str(path), 'dicom_id': path.stem})

df_all_files = pd.DataFrame(all_files_data)
print(f"   Total fisiere JPG gasite: {len(df_all_files)}")

# Imagini Fizice + Imagini ED
master_df = pd.merge(df_all_files, df_ed_images, on='dicom_id', how='inner')

# Adaugare Split
master_df = pd.merge(master_df, df_split, on='dicom_id', how='inner')

# Adaugare Label-uri (cele cu NaN unde era -1)
if 'study_id_y' in master_df.columns:
    master_df = master_df.drop(columns=['study_id_y']).rename(columns={'study_id_x': 'study_id'})

master_df = pd.merge(master_df, df_chexpert, on='study_id', how='inner')

# ============================================================
# 7. RAPORT SI SALVARE
# ============================================================
def print_label_statistics(df, labels):
    print("\n" + "=" * 60)
    print("STATISTICI ETICHETE (MASTER DATASET)")
    print("=" * 60)
    stats = []
    for label in labels:
        # dropna=False va numara si NaN-urile
        counts = df[label].value_counts(dropna=False)
        stats.append({
            'Patologie': label,
            'Pozitiv (1)': counts.get(1.0, 0),
            'Negativ (0)': counts.get(0.0, 0),
            'GOL (NaN/-1)': int(df[label].isna().sum()) # Aici vor aparea fostele -1
        })

    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    print("=" * 60)

print_label_statistics(master_df, pathology_cols)

output_filename = "master_dataset.csv"
master_df.to_csv(output_filename, index=False)
print(f"\nSUCCES: FISIER SALVAT: {output_filename}")
print(f"TOTAL IMAGINI IN DATASET FINAL: {len(master_df)}")
print("NOTA: Celulele goale din CSV sunt fostele valori de -1.")