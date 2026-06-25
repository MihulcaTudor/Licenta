import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer


def build_multimodal_dataset(edstays_path, triage_path, vitals_path, diagnosis_path, chexpert_path, metadata_path,
                             output_csv_path):
    print("1. Încărcarea datelor...")
    # Încărcăm doar coloanele strict necesare pentru a eficientiza consumul de memorie
    edstays = pd.read_csv(edstays_path,
                          usecols=['subject_id', 'hadm_id', 'stay_id', 'intime', 'gender', 'race', 'arrival_transport'])
    triage = pd.read_csv(triage_path, usecols=['subject_id', 'stay_id', 'chiefcomplaint', 'acuity', 'pain'])
    vitalsign = pd.read_csv(vitals_path)
    diagnosis = pd.read_csv(diagnosis_path, usecols=['subject_id', 'stay_id', 'icd_code'])
    chexpert = pd.read_csv(chexpert_path)
    metadata = pd.read_csv(metadata_path, usecols=['subject_id', 'study_id', 'StudyDate', 'StudyTime'])

    # ==========================================
    # PARTEA 1: PREPROCESARE DATE URGENTE (ED)
    # ==========================================
    print("2. Agregarea semnelor vitale (vitalsign)...")
    vitals_agg = vitalsign.groupby('stay_id').agg({
        'temperature': ['min', 'max', 'mean'],
        'heartrate': ['min', 'max', 'mean', 'std'],
        'resprate': ['min', 'max', 'mean'],
        'o2sat': ['min', 'mean'],
        'sbp': ['min', 'max', 'mean'],
        'dbp': ['min', 'max', 'mean']
    }).reset_index()

    # Aplatizăm coloanele MultiIndex (ex: temperature_min)
    vitals_agg.columns = ['stay_id'] + [f"{col[0]}_{col[1]}" for col in vitals_agg.columns.values[1:]]

    print("3. Fuziunea datelor de bază ED și curățarea demograficelor...")
    # Integrăm și variabilele demografice (gender, race, transport)
    df_ed = edstays.merge(
        triage, on=['subject_id', 'stay_id'], how='inner'
    ).merge(vitals_agg, on='stay_id', how='left')

    # Curățare text și durere
    df_ed['chiefcomplaint'] = df_ed['chiefcomplaint'].fillna('unknown')
    df_ed['chiefcomplaint'] = df_ed['chiefcomplaint'].apply(
        lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x).lower().strip()))

    df_ed['pain'] = pd.to_numeric(df_ed['pain'], errors='coerce').fillna(0)
    df_ed['gender'] = df_ed['gender'].fillna('unknown')
    df_ed['race'] = df_ed['race'].fillna('unknown')
    df_ed['arrival_transport'] = df_ed['arrival_transport'].fillna('unknown')

    # ==========================================
    # PARTEA 2: PREPROCESARE ETICHETE ICD (FAZA 1)
    # ==========================================
    print("4. Extragerea și binarizarea etichetelor ICD-10 (primele 3 caractere)...")
    diagnosis['icd_3'] = diagnosis['icd_code'].astype(str).str[:3]
    diag_grouped = diagnosis.groupby('stay_id')['icd_3'].apply(list).reset_index()

    # Folosim MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    icd_matrix = mlb.fit_transform(diag_grouped['icd_3'])

    icd_df = pd.DataFrame(icd_matrix, columns=[f"ICD_{c}" for c in mlb.classes_])
    icd_df['stay_id'] = diag_grouped['stay_id']

    # Adăugăm etichetele ICD la datele ED
    df_ed = df_ed.merge(icd_df, on='stay_id', how='inner')

    # ==========================================
    # PARTEA 3: ALINIEREA TEMPORALĂ CU CHEXPERT (OPTIMIZATĂ MEMORY-SAFE)
    # ==========================================
    print("5. Procesarea metadatelor și alinierea temporală a radiografiilor...")

    # Generăm timpul exact al radiografiei
    metadata['StudyDate'] = metadata['StudyDate'].astype(str).str.split('.').str[0]
    metadata['StudyTime'] = metadata['StudyTime'].astype(str).str.split('.').str[0].str.zfill(6)
    metadata['cxr_time'] = pd.to_datetime(metadata['StudyDate'] + ' ' + metadata['StudyTime'], format='%Y%m%d %H%M%S',
                                          errors='coerce')

    # Fuzionăm etichetele CheXpert cu timpul radiografiei
    chexpert_full = chexpert.merge(metadata[['subject_id', 'study_id', 'cxr_time']], on=['subject_id', 'study_id'],
                                   how='inner')

    # Formatăm timpul de internare din ED
    df_ed['intime'] = pd.to_datetime(df_ed['intime'], errors='coerce')

    print("6. Intersecția finală (Filtru ferastră 24h)...")

    # TRUC PENTRU RAM: Creăm un DataFrame foarte mic doar pentru alinierea temporală
    df_timeline = df_ed[['subject_id', 'stay_id', 'intime']]

    # JOIN pe pacient DOAR pe dataframe-ul mic (previne ArrayMemoryError de 38GB)
    df_merged_timeline = df_timeline.merge(chexpert_full, on='subject_id', how='inner')

    # Calculăm diferența (în ore) dintre Rx și momentul sosirii la UPU
    df_merged_timeline['time_diff_hours'] = (df_merged_timeline['cxr_time'] - df_merged_timeline[
        'intime']).dt.total_seconds() / 3600

    # Filtrăm radiografiile făcute între -2h și +24h
    df_valid_cxr = df_merged_timeline[
        (df_merged_timeline['time_diff_hours'] >= -2) & (df_merged_timeline['time_diff_hours'] <= 24)].copy()

    # Păstrăm radiografia cea mai apropiată de 'intime'
    df_valid_cxr['abs_time_diff'] = df_valid_cxr['time_diff_hours'].abs()
    df_best_cxr = df_valid_cxr.sort_values('abs_time_diff').drop_duplicates(subset=['stay_id'], keep='first')

    # Curățăm coloanele temporale inutile din tabelul de Rx rezultat
    df_best_cxr = df_best_cxr.drop(columns=['time_diff_hours', 'abs_time_diff', 'cxr_time', 'intime', 'subject_id'])

    # ACUM facem join-ul final 1-la-1 cu tabelul mamut `df_ed` folosind `stay_id`
    df_final = df_ed.merge(df_best_cxr, on='stay_id', how='inner')

    print("7. Curățarea etichetelor CheXpert și a valorilor lipsă...")
    chexpert_labels = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    # Curățare CheXpert (NaN -> 0, -1 (incert) -> 1)
    for col in chexpert_labels:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0).replace(-1, 1).astype(int)

    # Imputare finală simplă pentru semnele vitale lipsă
    numeric_vitals = [c for c in df_final.columns if 'min' in c or 'max' in c or 'mean' in c or 'std' in c]
    df_final[numeric_vitals] = df_final[numeric_vitals].fillna(df_final[numeric_vitals].mean())

    # Curățenie finală de coloane inutile
    df_final = df_final.drop(columns=['intime', 'hadm_id'])

    print(f"Dataset construit cu succes! Total vizite valide și corect aliniate: {df_final.shape[0]}")

    # Salvare în CSV
    df_final.to_csv(output_csv_path, index=False)
    print(f"Fișier salvat: {output_csv_path}")

    return df_final, mlb.classes_


# ==========================================
# CĂILE CĂTRE FIȘIERELE TALE MIMIC:
# ==========================================
if __name__ == "__main__":
    EDSTAYS = r"D:\MIMIC\MIMIC-IV-ED\edstays.csv"
    TRIAGE = r"D:\MIMIC\MIMIC-IV-ED\triage.csv"
    VITALS = r"D:\MIMIC\MIMIC-IV-ED\vitalsign.csv"
    DIAGNOSIS = r"D:\MIMIC\MIMIC-IV-ED\diagnosis.csv"
    CHEXPERT = r"D:\MIMIC\MIMIC-CXR-JPG\mimic-cxr-2.0.0-chexpert.csv"

    # ATENȚIE: Adaugă calea către fișierul de metadate!
    METADATA = r"D:\MIMIC\MIMIC-CXR-JPG\mimic-cxr-2.0.0-metadata.csv"

    OUTPUT = r"C:\Users\2D\PycharmProjects\licenta_pt\dataset_multimodal_final.csv"

    df_final, icd_classes = build_multimodal_dataset(EDSTAYS, TRIAGE, VITALS, DIAGNOSIS, CHEXPERT, METADATA, OUTPUT)