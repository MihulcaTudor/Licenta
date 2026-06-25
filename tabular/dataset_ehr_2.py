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

# Tabelul Patients (MIMIC-IV Core) - Adaugat conform structurii tale
PATIENTS_DIR = ROOT_DIR / "MIMIC-IV" / "core"
patients_csv_path = PATIENTS_DIR / "patients.csv"

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
    ("MEDRECON", medrecon_csv_path),
    ("PATIENTS", patients_csv_path)
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
    if pd.isna(value): return None

    if name == 'pain':
        if isinstance(value, str) and not value.strip().replace('.', '', 1).isdigit():
            clean_str = str(value).strip().lower()
            if 'asleep' in clean_str: return "PAIN_ASLEEP"
            elif 'unable' in clean_str: return "PAIN_UNABLE"
            elif 'refus' in clean_str: return "PAIN_REFUSED"
            else: return "PAIN_UNKNOWN"

    try: val = float(value)
    except ValueError: return None

    if name == 'heartrate':
        return "HR_LOW" if val < 60 else "HR_HIGH" if val > 100 else "HR_NORMAL"
    elif name == 'o2sat':
        return "O2_CRITICAL" if val < 90 else "O2_LOW" if val < 95 else "O2_NORMAL"
    elif name == 'temperature':
        return "TEMP_LOW" if val < 95.0 else "TEMP_FEVER" if val > 100.4 else "TEMP_NORMAL"
    elif name == 'sbp':
        return "SBP_LOW" if val < 90 else "SBP_HIGH" if val > 140 else "SBP_NORMAL"
    elif name == 'resprate':
        return "RR_LOW" if val < 12 else "RR_HIGH" if val > 20 else "RR_NORMAL"
    elif name == 'pain':
        try:
            p = int(val)
            return "PAIN_NONE" if p == 0 else "PAIN_MILD" if p < 5 else "PAIN_SEVERE"
        except: return "PAIN_UNKNOWN"
    return None

def clean_med_name(name):
    if pd.isna(name): return "UNKNOWN_MED"
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', str(name).strip().upper())
    return re.sub(r'_+', '_', clean_name).strip('_')

# ============================================================
# 4. ÎNCĂRCARE, SINCRONIZARE ȘI CALCUL VÂRSTĂ
# ============================================================
print("\n--- ÎNCEPE PROCESAREA DATELOR ---")
df_metadata = pd.read_csv(metadata_csv_path, usecols=['subject_id', 'study_id', 'StudyDate', 'StudyTime'])
df_edstays = pd.read_csv(edstays_csv_path, usecols=['subject_id', 'stay_id', 'intime', 'outtime'])
df_patients = pd.read_csv(patients_csv_path, usecols=['subject_id', 'anchor_age', 'anchor_year'])

# Formatare timp radiografii
date_str = df_metadata['StudyDate'].astype(float).astype(int).astype(str)
time_str = df_metadata['StudyTime'].astype(float).astype(int).astype(str).str.zfill(6)
df_metadata['study_datetime'] = pd.to_datetime(date_str + " " + time_str, format='%Y%m%d %H%M%S', errors='coerce')

# Formatare timp vizite ED
df_edstays['intime'] = pd.to_datetime(df_edstays['intime'], errors='coerce')
df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'], errors='coerce')

# Calculare vârstă exactă la momentul vizitei ED
df_edstays['intime_year'] = df_edstays['intime'].dt.year

# --- MODIFICARE AICI: inner join pentru a garanta pacienții ED compleți ---
df_edstays = pd.merge(df_edstays, df_patients, on='subject_id', how='inner')

# Acum suntem siguri că datele există, nu va mai da NaN
df_edstays['computed_age'] = df_edstays['anchor_age'] + (df_edstays['intime_year'] - df_edstays['anchor_year'])
# Ne asigurăm că nu avem valori negative sau aberante
df_edstays['computed_age'] = df_edstays['computed_age'].clip(lower=0).fillna(0).astype(int)

df_merged = pd.merge(df_metadata, df_edstays, on='subject_id', how='inner')
df_ed_studies = df_merged.query('study_datetime >= intime and study_datetime <= outtime').copy()
df_ed_studies = df_ed_studies.drop_duplicates(subset=['study_id'])
valid_stay_ids = df_ed_studies['stay_id'].unique()

print(f"   Studii radiologice valide: {len(df_ed_studies)}")
print(f"   Episoade ED (stay_id) unice valide: {len(valid_stay_ids)}")
# ============================================================
# 5. PREPROCESARE ETICHETE CHEXPERT
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

# Dicționare pentru acces rapid
intime_map = df_edstays.set_index('stay_id')['intime'].to_dict()
age_map = df_edstays.set_index('stay_id')['computed_age'].to_dict()

sequences = []
vital_columns = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'pain']

# Definire Token Types (pot fi ajustate in PyTorch)
TYPE_CLS = 0
TYPE_VS = 1
TYPE_VE = 2
TYPE_REG = 3
TYPE_LAB = 4 # Pentru Vitals/Triaj (asemănător L)
TYPE_MED = 5 # Pentru Medicație (asemănător M)

for stay_id in valid_stay_ids:
    intime = intime_map.get(stay_id)
    age = age_map.get(stay_id, 0) # Vârsta este deja int și sigură
    if pd.isna(intime): continue
    clinical_events = []

    # Triaj (Timp=0)
    triage_data = df_triage[df_triage['stay_id'] == stay_id]
    if not triage_data.empty:
        row = triage_data.iloc[0]
        if not pd.isna(row['acuity']):
            clinical_events.append((0, f"ACUITY_{int(row['acuity'])}", TYPE_LAB))
        for col in vital_columns:
            token = bin_vital(col, row[col])
            if token: clinical_events.append((0, token, TYPE_LAB))

    # Medicația de Acasă (Timp=0)
    medrecon_data = df_medrecon[df_medrecon['stay_id'] == stay_id]
    for _, row in medrecon_data.iterrows():
        if not pd.isna(row['name']):
            med_token = f"MED_HOME_{clean_med_name(row['name'])}"
            clinical_events.append((0, med_token, TYPE_MED))

    # Semne Vitale
    vitals_data = df_vitals[df_vitals['stay_id'] == stay_id]
    for _, row in vitals_data.iterrows():
        if pd.isna(row['charttime']): continue
        delta_mins = int((row['charttime'] - intime).total_seconds() / 60.0)
        if delta_mins < -60: continue # Ignorăm anomaliile de dinainte de internare
        for col in vital_columns:
            token = bin_vital(col, row[col])
            if token: clinical_events.append((delta_mins, token, TYPE_LAB))

    # Medicația în ED
    pyxis_data = df_pyxis[df_pyxis['stay_id'] == stay_id]
    for _, row in pyxis_data.iterrows():
        if pd.isna(row['charttime']) or pd.isna(row['name']): continue
        delta_mins = int((row['charttime'] - intime).total_seconds() / 60.0)
        if delta_mins < -60: continue
        med_token = f"MED_ED_{clean_med_name(row['name'])}"
        clinical_events.append((delta_mins, med_token, TYPE_MED))

    if not clinical_events:
        continue

    clinical_events.sort(key=lambda x: x[0])

    # Construire Liste Paralele conform Figurii 1 din EHRMAMBA
    concept_list = ["[CLS]", "[VS]"]
    time_list = [0, 0] # CLS are time 0. VS are time 0 (start vizită).
    type_list = [TYPE_CLS, TYPE_VS]
    age_list = [0, age] # CLS are Age 0. VS preia vârsta.
    segment_list = [0, 1] # CLS are Segment 0. Vizita e Segment 1.
    visit_order_list = [0, 1]

    for ev_time, ev_token, ev_type in clinical_events:
        concept_list.append(ev_token)
        time_list.append(max(0, ev_time)) # Timpul este în minute
        type_list.append(ev_type)
        age_list.append(age)
        segment_list.append(1)
        visit_order_list.append(1)

    last_time = time_list[-1] if len(time_list) > 2 else 0
    concept_list.extend(["[VE]", "[REG]"])
    time_list.extend([last_time, last_time])
    type_list.extend([TYPE_VE, TYPE_REG])
    age_list.extend([age, age])
    segment_list.extend([1, 1])
    visit_order_list.extend([1, 1])

    seq_len = len(concept_list)
    position_list = list(range(seq_len))

    sequences.append({
        'stay_id': stay_id,
        'concept_seq': ",".join(concept_list),
        'time_seq': ",".join(map(str, time_list)),
        'type_seq': ",".join(map(str, type_list)),
        'position_seq': ",".join(map(str, position_list)),
        'age_seq': ",".join(map(str, age_list)),
        'segment_seq': ",".join(map(str, segment_list)),
        'visit_order_seq': ",".join(map(str, visit_order_list)),
        'seq_length': seq_len
    })

df_sequences = pd.DataFrame(sequences)

# ============================================================
# 7. MERGE FINAL ȘI SALVARE
# ============================================================
df_keys = df_ed_studies[['subject_id', 'stay_id', 'study_id']]
master_df = pd.merge(df_keys, df_sequences, on='stay_id', how='inner')
master_df = pd.merge(master_df, df_chexpert, on='study_id', how='inner')

output_filename = "ehrmamba_ed_dataset_2.csv"
master_df.to_csv(output_filename, index=False)

print("\n" + "=" * 50)
print(f"SUCCES: FIȘIER EHRMAMBA SALVAT CA: {output_filename}")
print(f"TOTAL SECVENȚE PROCESATE: {len(master_df)}")
print("=" * 50)