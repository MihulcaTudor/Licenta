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

# Fisierele din MIMIC-CXR-JPG
CXR_JPG_DIR = ROOT_DIR / "MIMIC-CXR-JPG"
metadata_csv_path = CXR_JPG_DIR / "mimic-cxr-2.0.0-metadata.csv"
chexpert_csv_path = CXR_JPG_DIR / "mimic-cxr-2.0.0-chexpert.csv"

# Fisierele din Urgențe (MIMIC-IV-ED)
ED_DIR = ROOT_DIR / "MIMIC-IV-ED"
edstays_csv_path = ED_DIR / "edstays.csv"
triage_csv_path = ED_DIR / "triage.csv"
vitalsign_csv_path = ED_DIR / "vitalsign.csv"
pyxis_csv_path = ED_DIR / "pyxis.csv"
medrecon_csv_path = ED_DIR / "medrecon.csv"

# ============================================================
# 2. VERIFICARE PRELIMINARĂ
# ============================================================
print(f"--- VERIFICARE CĂI (SISTEM: {os.name}) ---")
files_to_check = [
    ("METADATA", metadata_csv_path),
    ("CHEXPERT", chexpert_csv_path),
    ("ED STAYS", edstays_csv_path),
    ("TRIAGE", triage_csv_path),
    ("VITALSIGN", vitalsign_csv_path),
    ("PYXIS", pyxis_csv_path),
    ("MEDRECON", medrecon_csv_path)
]

paths_ok = True
for name, p in files_to_check:
    if not p.exists():
        print(f"EROARE: NU GĂSESC {name} LA: {p}")
        paths_ok = False
    else:
        print(f"CONFIRMAT: GĂSIT {name}")

if not paths_ok:
    print("SISTEMUL SE OPREȘTE. VERIFICĂ CĂILE.")
    sys.exit()


# ============================================================
# 3. FUNCȚII DE DISCRETIZARE ȘI CURĂȚARE
# ============================================================
def bin_vital(name, value):
    """Transformă o valoare numerică sau text într-un token discret."""
    if pd.isna(value):
        return None

    # Tratăm cazurile textuale frecvente în coloana 'pain'
    if name == 'pain':
        if isinstance(value, str) and not value.strip().replace('.', '', 1).isdigit():
            clean_str = str(value).strip().lower()
            if 'asleep' in clean_str:
                return "PAIN_ASLEEP"
            elif 'unable' in clean_str:
                return "PAIN_UNABLE"
            elif 'refus' in clean_str:
                return "PAIN_REFUSED"
            else:
                return "PAIN_UNKNOWN"

    try:
        val = float(value)
    except ValueError:
        return None  # Ignorăm valorile textuale corupte din alte coloane

    if name == 'heartrate':
        if val < 60:
            return "HR_LOW"
        elif val > 100:
            return "HR_HIGH"
        else:
            return "HR_NORMAL"
    elif name == 'o2sat':
        if val < 90:
            return "O2_CRITICAL"
        elif val < 95:
            return "O2_LOW"
        else:
            return "O2_NORMAL"
    elif name == 'temperature':
        if val < 95.0:
            return "TEMP_LOW"
        elif val > 100.4:
            return "TEMP_FEVER"
        else:
            return "TEMP_NORMAL"
    elif name == 'sbp':
        if val < 90:
            return "SBP_LOW"
        elif val > 140:
            return "SBP_HIGH"
        else:
            return "SBP_NORMAL"
    elif name == 'resprate':
        if val < 12:
            return "RR_LOW"
        elif val > 20:
            return "RR_HIGH"
        else:
            return "RR_NORMAL"
    elif name == 'pain':
        try:
            p = int(val)
            if p == 0:
                return "PAIN_NONE"
            elif p < 5:
                return "PAIN_MILD"
            else:
                return "PAIN_SEVERE"
        except:
            return "PAIN_UNKNOWN"

    return None


def clean_med_name(name):
    """Curăță numele medicamentului."""
    if pd.isna(name): return "UNKNOWN_MED"
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', str(name).strip().upper())
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    return clean_name


# ============================================================
# 4. ÎNCĂRCARE, SINCRONIZARE ED ȘI FILTRARE
# ============================================================
print("\n--- ÎNCEPE PROCESAREA DATELOR ---")
df_metadata = pd.read_csv(metadata_csv_path, usecols=['subject_id', 'study_id', 'StudyDate', 'StudyTime'])
df_edstays = pd.read_csv(edstays_csv_path, usecols=['subject_id', 'stay_id', 'intime', 'outtime'])

date_str = df_metadata['StudyDate'].astype(float).astype(int).astype(str)
time_str = df_metadata['StudyTime'].astype(float).astype(int).astype(str).str.zfill(6)
df_metadata['study_datetime'] = pd.to_datetime(date_str + " " + time_str, format='%Y%m%d %H%M%S', errors='coerce')

df_edstays['intime'] = pd.to_datetime(df_edstays['intime'], errors='coerce')
df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'], errors='coerce')

# Fuziune și filtrare
df_merged = pd.merge(df_metadata, df_edstays, on='subject_id', how='inner')
df_ed_studies = df_merged.query('study_datetime >= intime and study_datetime <= outtime').copy()
df_ed_studies = df_ed_studies.drop_duplicates(subset=['study_id'])
valid_stay_ids = df_ed_studies['stay_id'].unique()
print(f"   Studii radiologice valide: {len(df_ed_studies)}")
print(f"   Episoade ED (stay_id) unice valide: {len(valid_stay_ids)}")

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
df_chexpert[pathology_cols] = df_chexpert[pathology_cols].fillna(0.0).replace(-1.0, 0.0)
df_chexpert = df_chexpert.drop_duplicates(subset=['study_id'])

# ============================================================
# 6. CONSTRUIREA SECVENȚELOR PARALELE (STANDARD EHRMAMBA)
# ============================================================
print("\n--- CONSTRUIRE SECVENȚE EHRMAMBA ---")

df_triage = pd.read_csv(triage_csv_path)
df_triage = df_triage[df_triage['stay_id'].isin(valid_stay_ids)]

df_vitals = pd.read_csv(vitalsign_csv_path)
df_vitals = df_vitals[df_vitals['stay_id'].isin(valid_stay_ids)]
df_vitals['charttime'] = pd.to_datetime(df_vitals['charttime'], errors='coerce')

df_pyxis = pd.read_csv(pyxis_csv_path)
df_pyxis = df_pyxis[df_pyxis['stay_id'].isin(valid_stay_ids)]
df_pyxis['charttime'] = pd.to_datetime(df_pyxis['charttime'], errors='coerce')

df_medrecon = pd.read_csv(medrecon_csv_path)
df_medrecon = df_medrecon[df_medrecon['stay_id'].isin(valid_stay_ids)]

intime_map = df_edstays.set_index('stay_id')['intime'].to_dict()

sequences = []
vital_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'pain']

# Type codings: 0 = Special, 1 = Vitals/Measurements, 2 = Medications
for stay_id in valid_stay_ids:
    intime = intime_map.get(stay_id)
    if pd.isna(intime): continue

    clinical_events = []  # Vom stoca (timp_relativ_minute, token, tip)

    # --- Triaj (Timpul 0, Type 1) ---
    triage_data = df_triage[df_triage['stay_id'] == stay_id]
    if not triage_data.empty:
        row = triage_data.iloc[0]
        if not pd.isna(row['acuity']):
            clinical_events.append((0, f"ACUITY_{int(row['acuity'])}", 1))
        for col in vital_columns:
            token = bin_vital(col, row[col])
            if token: clinical_events.append((0, token, 1))

    # --- Medicația de Acasă (Timpul 0, Type 2) ---
    medrecon_data = df_medrecon[df_medrecon['stay_id'] == stay_id]
    for _, row in medrecon_data.iterrows():
        if not pd.isna(row['name']):
            med_token = f"MED_HOME_{clean_med_name(row['name'])}"
            clinical_events.append((0, med_token, 2))

    # --- Monitorizare Semne Vitale (Timpul relativ, Type 1) ---
    vitals_data = df_vitals[df_vitals['stay_id'] == stay_id]
    for _, row in vitals_data.iterrows():
        if pd.isna(row['charttime']): continue
        delta_mins = int((row['charttime'] - intime).total_seconds() / 60.0)
        if delta_mins < -60: continue
        for col in vital_columns:
            token = bin_vital(col, row[col])
            if token: clinical_events.append((delta_mins, token, 1))

    # --- Medicația administrată în ED (Timpul relativ, Type 2) ---
    pyxis_data = df_pyxis[df_pyxis['stay_id'] == stay_id]
    for _, row in pyxis_data.iterrows():
        if pd.isna(row['charttime']) or pd.isna(row['name']): continue
        delta_mins = int((row['charttime'] - intime).total_seconds() / 60.0)
        if delta_mins < -60: continue
        med_token = f"MED_ED_{clean_med_name(row['name'])}"
        clinical_events.append((delta_mins, med_token, 2))

    if not clinical_events:
        continue

    # Sortăm evenimentele clinice cronologic
    clinical_events.sort(key=lambda x: x[0])

    # Construim listele paralele cu tokenii speciali
    # 1. Start Tokens
    concept_list = ["[CLS]", "[VS]"]
    time_list = [0, 0]
    type_list = [0, 0]

    # 2. Clinical Events
    for ev_time, ev_token, ev_type in clinical_events:
        concept_list.append(ev_token)
        time_list.append(max(0, ev_time))  # Timpul nu ar trebui sa fie negativ
        type_list.append(ev_type)

    # 3. End Tokens
    last_time = time_list[-1] if len(time_list) > 2 else 0
    concept_list.extend(["[VE]", "[REG]"])
    time_list.extend([last_time, last_time])
    type_list.extend([0, 0])

    # 4. Position Array
    position_list = list(range(len(concept_list)))

    # Transformăm listele în string-uri separate prin virgulă pentru stocare în CSV
    sequences.append({
        'stay_id': stay_id,
        'concept_seq': ",".join(concept_list),
        'time_seq': ",".join(map(str, time_list)),
        'type_seq': ",".join(map(str, type_list)),
        'position_seq': ",".join(map(str, position_list)),
        'seq_length': len(concept_list)
    })

df_sequences = pd.DataFrame(sequences)

# ============================================================
# 7. MERGE FINAL (SUBJECT_ID + LABELS) ȘI SALVARE
# ============================================================
# Adăugăm subject_id și study_id pentru mapare completă
df_keys = df_ed_studies[['subject_id', 'stay_id', 'study_id']]
master_df = pd.merge(df_keys, df_sequences, on='stay_id', how='inner')
master_df = pd.merge(master_df, df_chexpert, on='study_id', how='inner')

output_filename = "ehrmamba_ed_dataset.csv"
master_df.to_csv(output_filename, index=False)

print("\n" + "=" * 50)
print(f"SUCCES: FIȘIER EHRMAMBA SALVAT CA: {output_filename}")
print(f"TOTAL SECVENȚE PROCESATE: {len(master_df)}")
print("=" * 50)