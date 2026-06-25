import pandas as pd
import numpy as np
import os

# # ==========================================
# # 1. CONFIGURAREA CĂILOR CĂTRE FIȘIERELE MIMIC
# # ==========================================
# MIMIC_ED_DIR = 'D:/MIMIC/MIMIC-IV-ED'
# MIMIC_HOSP_DIR = 'D:/MIMIC/mimic-iv-3.1/hosp'
#
# NEW_TABULAR_DATASET_PATH = 'cehr_bert_longitudinal_dataset.csv'
#
# # Cele 14 patologii țintă
# TARGET_PATHOLOGIES = [
#     'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
#     'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
#     'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
#     'Support Devices', 'No Finding'
# ]
#
# # ==========================================
# # 2. DEFINIREA INTERVALELOR CLINICE (BINNING)
# # ==========================================
# BINS = {
#     'HR': ([0, 50, 60, 100, 120, 300], ['VERY_LOW', 'LOW', 'NORMAL', 'HIGH', 'VERY_HIGH']),
#     'SpO2': ([0, 88, 92, 95, 100], ['CRITICAL', 'LOW', 'BORDERLINE', 'NORMAL']),
#     'TEMP': ([0, 95, 97.7, 99.5, 100.4, 104, 150], ['HYPOTHERMIA', 'LOW', 'NORMAL', 'ELEVATED', 'FEVER', 'HIGH_FEVER']),
#     'SBP': (
#     [0, 90, 120, 130, 140, 180, 300], ['HYPOTENSION', 'NORMAL', 'ELEVATED', 'STAGE1_HTN', 'STAGE2_HTN', 'CRISIS']),
#     'DBP': ([0, 60, 80, 90, 120, 200], ['HYPOTENSION', 'NORMAL', 'STAGE1_HTN', 'STAGE2_HTN', 'CRISIS']),
#     'RR': ([0, 8, 12, 20, 24, 100], ['CRITICAL_LOW', 'LOW', 'NORMAL', 'HIGH', 'CRITICAL_HIGH']),
#     'AGE': ([18, 30, 45, 65, 80, 120], ['YOUNG_ADULT', 'ADULT', 'MIDDLE_AGE', 'SENIOR', 'ELDERLY'])
# }
#
#
# def discretize_val(val, prefix, config):
#     if pd.isna(val):
#         return ""  # Nu adăugăm token dacă lipsește măsurătoarea
#     try:
#         bin_label = pd.cut([val], bins=config[0], labels=config[1], include_lowest=True)[0]
#         if pd.isna(bin_label): return ""
#         return f"[{prefix}_{bin_label}]"
#     except:
#         return ""
#
#
# def main():
#     print("1. Încărcarea datelor (Triage, Patients, Vitalsign, Pyxis)...")
#     triage_df = pd.read_csv(os.path.join(MIMIC_ED_DIR, 'triage.csv'))
#     vitals_df = pd.read_csv(os.path.join(MIMIC_ED_DIR, 'vitalsign.csv'))
#     pyxis_df = pd.read_csv(os.path.join(MIMIC_ED_DIR, 'pyxis.csv'))
#     patients_df = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, 'patients.csv'))
#
#     print("2. Procesarea contextului de bază (Triage & Demografice)...")
#     base_df = pd.merge(triage_df, patients_df[['subject_id', 'anchor_age', 'gender']], on='subject_id', how='left')
#
#     base_sequences = []
#     for _, row in base_df.iterrows():
#         tokens = []
#         if pd.notna(row['gender']): tokens.append(f"[GENDER_{row['gender']}]")
#
#         age_t = discretize_val(row['anchor_age'], 'AGE', BINS['AGE'])
#         if age_t: tokens.append(age_t)
#
#         # Prefixăm cu TRIAGE_ pentru a ști că sunt valorile inițiale
#         for col, prefix in [('heartrate', 'TRIAGE_HR'), ('o2sat', 'TRIAGE_SpO2'), ('temperature', 'TRIAGE_TEMP'),
#                             ('sbp', 'TRIAGE_SBP'), ('dbp', 'TRIAGE_DBP'), ('resprate', 'TRIAGE_RR')]:
#             t = discretize_val(row[col], prefix, BINS[prefix.split('_')[1]])
#             if t: tokens.append(t)
#
#         base_sequences.append({
#             'stay_id': row['stay_id'],
#             'subject_id': row['subject_id'],
#             'base_seq': " ".join(tokens)
#         })
#     base_seq_df = pd.DataFrame(base_sequences)
#
#     print("3. Procesarea evenimentelor longitudinale (Semne vitale continue)...")
#     # Generăm o secvență de tokeni pentru fiecare moment de timp (charttime) în care s-au luat vitale
#     vitals_events = []
#     for _, row in vitals_df.dropna(subset=['charttime']).iterrows():
#         tokens = []
#         for col, prefix in [('heartrate', 'HR'), ('o2sat', 'SpO2'), ('temperature', 'TEMP'),
#                             ('sbp', 'SBP'), ('dbp', 'DBP'), ('resprate', 'RR')]:
#             t = discretize_val(row[col], prefix, BINS[prefix])
#             if t: tokens.append(t)
#
#         if tokens:
#             vitals_events.append({
#                 'stay_id': row['stay_id'],
#                 'charttime': row['charttime'],
#                 'event_tokens': " ".join(tokens)
#             })
#     vitals_events_df = pd.DataFrame(vitals_events)
#
#     print("4. Procesarea evenimentelor medicamentoase (Pyxis)...")
#     pyxis_df = pyxis_df.dropna(subset=['charttime', 'name'])
#     # Curățăm numele medicamentului (înlocuim spațiile cu underscore, punem uppercase)
#     # Ex: "Aspirin 81mg" devine "[MED_ASPIRIN_81MG]"
#     pyxis_df['event_tokens'] = pyxis_df['name'].astype(str).str.replace(' ', '_').str.upper()
#     pyxis_df['event_tokens'] = "[MED_" + pyxis_df['event_tokens'] + "]"
#     med_events_df = pyxis_df[['stay_id', 'charttime', 'event_tokens']].copy()
#
#     print("5. Fuziunea și ordonarea cronologică a tuturor evenimentelor...")
#     # Combinăm vitalele și medicamentele
#     all_events_df = pd.concat([vitals_events_df, med_events_df])
#     # Ordonăm cronologic după pacient și timpul evenimentului
#     all_events_df = all_events_df.sort_values(by=['stay_id', 'charttime'])
#
#     # Grupăm toate evenimentele unui stay_id într-un singur șir de text separat prin spațiu
#     timeline_df = all_events_df.groupby('stay_id')['event_tokens'].apply(lambda x: ' '.join(x)).reset_index()
#     timeline_df.rename(columns={'event_tokens': 'timeline_seq'}, inplace=True)
#
#     print("6. Construcția Dataset-ului Final...")
#     # Unim Contextul de Bază (Triaj) cu Timeline-ul cronologic
#     final_df = pd.merge(base_seq_df, timeline_df, on='stay_id', how='left')
#
#     # Dacă un pacient nu a avut evenimente după triaj, înlocuim NaN cu string gol
#     final_df['timeline_seq'] = final_df['timeline_seq'].fillna('')
#
#     # Secvența finală care va intra în BERT: Triaj + [SEP] + Timeline
#     final_df['text_sequence'] = final_df['base_seq'] + " [SEP] " + final_df['timeline_seq']
#
#     # Adăugăm coloanele goale necesare arhitecturii (study_id + 14 patologii)
#     final_df['study_id'] = np.nan
#     for pathology in TARGET_PATHOLOGIES:
#         final_df[pathology] = np.nan
#
#     final_columns = ['subject_id', 'stay_id', 'study_id', 'text_sequence'] + TARGET_PATHOLOGIES
#     final_df = final_df[final_columns]
#
#     final_df.to_csv(NEW_TABULAR_DATASET_PATH, index=False)
#     print(f"\n[SUCCES] Dataset creat! -> {NEW_TABULAR_DATASET_PATH}")
#
#     # Previzualizare pentru un pacient
#     sample = final_df[final_df['timeline_seq'] != ''].iloc[0]
#     print("\n[EXEMPLU SECVENȚĂ PENTRU REȚEA]:")
#     print(sample['text_sequence'][:500] + " ... (trunchiat)")
#
#
# if __name__ == "__main__":
#     main()
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURARE CĂI FIȘIERE SURSĂ
# ==========================================
TEXT_DATASET_PATH = 'cehr_bert_longitudinal_dataset.csv'  # Dataset-ul nostru text
EDSTAYS_CSV_PATH = 'D:/MIMIC/MIMIC-IV-ED/edstays.csv'
CXR_META_CSV_PATH = 'D:/MIMIC/MIMIC-CXR-JPG/mimic-cxr-2.0.0-metadata.csv'
CHEXPERT_CSV_PATH = 'D:/MIMIC/MIMIC-CXR-JPG/mimic-cxr-2.0.0-chexpert.csv'

OUTPUT_DATASET_PATH = 'cehr_bert_ready_to_train.csv'

TARGET_PATHOLOGIES = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices'
]


def main():
    print("1. Încărcare dataset textual...")
    text_df = pd.read_csv(TEXT_DATASET_PATH)
    # Eliminăm coloanele goale pentru a face loc celor reale
    cols_to_drop = TARGET_PATHOLOGIES + ['No Finding', 'study_id']
    text_df = text_df.drop(columns=[col for col in cols_to_drop if col in text_df.columns])

    print("2. Prelucrare timp din edstays (UPU)...")
    edstays = pd.read_csv(EDSTAYS_CSV_PATH, usecols=['subject_id', 'stay_id', 'intime', 'outtime'])
    edstays['intime'] = pd.to_datetime(edstays['intime'])
    edstays['outtime'] = pd.to_datetime(edstays['outtime'])

    print("3. Prelucrare timp din metadatele radiologice (CXR)...")
    cxr_meta = pd.read_csv(CXR_META_CSV_PATH, usecols=['subject_id', 'study_id', 'StudyDate', 'StudyTime'])

    # Formatăm StudyDate și StudyTime într-un singur obiect datetime (T_scan)
    # StudyDate este de tip int (ex: 21100115), StudyTime are fractiuni (ex: 120536.23)
    cxr_meta['StudyDate'] = cxr_meta['StudyDate'].astype(str)
    # Adăugăm zerouri la StudyTime pentru a avea formatul corect HHMMSS
    cxr_meta['StudyTime'] = cxr_meta['StudyTime'].astype(str).apply(lambda x: x.split('.')[0].zfill(6))
    cxr_meta['T_scan'] = pd.to_datetime(cxr_meta['StudyDate'] + cxr_meta['StudyTime'], format='%Y%m%d%H%M%S',
                                        errors='coerce')
    cxr_meta = cxr_meta.dropna(subset=['T_scan'])

    print("4. Efectuăm Sincronizarea Temporală (Inner Merge + Filtru Temporal)...")
    # Facem join pe pacient
    merged_times = pd.merge(edstays, cxr_meta, on='subject_id', how='inner')

    # Păstrăm DOAR radiografiile făcute ÎN TIMPUL vizitei la urgențe
    # T_intime <= T_scan <= T_outtime
    valid_scans = merged_times[(merged_times['T_scan'] >= merged_times['intime']) &
                               (merged_times['T_scan'] <= merged_times['outtime'])]

    # Avem acum corespondența perfectă: stay_id -> study_id
    stay_to_study = valid_scans[['subject_id', 'stay_id', 'study_id']]

    print("5. Citim și procesăm etichetele CheXpert...")
    chexpert = pd.read_csv(CHEXPERT_CSV_PATH, usecols=['subject_id', 'study_id'] + TARGET_PATHOLOGIES)

    # Aplicăm strategia U-Zeros (-1.0 -> 0.0, NaN -> 0.0)
    for pathology in TARGET_PATHOLOGIES:
        chexpert[pathology] = chexpert[pathology].replace(-1.0, 0.0).fillna(0.0).astype(int)

    print("6. Asociem etichetele CheXpert cu vizitele ED...")
    labels_mapped = pd.merge(stay_to_study, chexpert, on=['subject_id', 'study_id'], how='inner')

    # Dacă un pacient a făcut 2 radiografii în aceeași vizită ED, luăm max-ul (agregare)
    # Astfel, dacă la a doua radiografie s-a văzut Pneumonia (1), vizita va fi etichetată cu 1.
    final_labels = labels_mapped.groupby(['subject_id', 'stay_id'])[TARGET_PATHOLOGIES].max().reset_index()

    print("7. Generarea Dataset-ului Final pentru CEHR-BERT...")
    # Lipim etichetele calculate de secvențele textuale
    final_dataset = pd.merge(text_df, final_labels, on=['subject_id', 'stay_id'], how='inner')

    final_dataset.to_csv(OUTPUT_DATASET_PATH, index=False)

    print(f"\n[SUCCES] Dataset salvat: {OUTPUT_DATASET_PATH}")
    print(f"Număr final de vizite UPU (stay_id) cu secvență text și radiografie validă: {len(final_dataset)}")

    print("\nPrevizualizare (primele 3 rânduri):")
    print(final_dataset[['stay_id', 'Pneumonia', 'Edema', 'Atelectasis']].head(3))


if __name__ == "__main__":
    main()