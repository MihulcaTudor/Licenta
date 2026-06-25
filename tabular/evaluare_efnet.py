import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Setări reproductibilitate
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 1. CLASELE NECESARE (Copiate din antrenament)
# ==========================================
class MultimodalMIMICDataset(Dataset):
    def __init__(self, df, text_col, num_cols, icd_cols, chexpert_cols, vocab=None, max_seq_len=None, scaler=None,
                 imputer=None):
        self.icd_labels = torch.tensor(df[icd_cols].fillna(0).values, dtype=torch.float32)
        self.chexpert_labels = torch.tensor(df[chexpert_cols].fillna(0).values, dtype=torch.float32)

        if imputer is None:
            self.imputer = SimpleImputer(strategy='mean')
            num_data_imputed = self.imputer.fit_transform(df[num_cols])
        else:
            self.imputer = imputer
            num_data_imputed = self.imputer.transform(df[num_cols])

        if scaler is None:
            self.scaler = StandardScaler()
            num_data = self.scaler.fit_transform(num_data_imputed)
        else:
            self.scaler = scaler
            num_data = self.scaler.transform(num_data_imputed)

        num_data = np.nan_to_num(num_data, nan=0.0)
        self.num_features = torch.tensor(num_data, dtype=torch.float32)

        self.text_data = df[text_col].fillna("").astype(str).tolist()

        if vocab is None:
            self.vocab, self.max_seq_len = self._build_vocab(self.text_data)
        else:
            self.vocab = vocab
            self.max_seq_len = max_seq_len

        self.text_features = self._tokenize_and_pad(self.text_data)

    def _build_vocab(self, texts):
        words = " ".join(texts).split()
        counter = Counter(words)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        for word, count in counter.items():
            if count > 1:
                vocab[word] = idx
                idx += 1
        max_len = max(len(text.split()) for text in texts)
        max_len = min(max_len, 20)
        return vocab, max_len

    def _tokenize_and_pad(self, texts):
        tokenized = []
        for text in texts:
            tokens = [self.vocab.get(w, self.vocab['<UNK>']) for w in text.split()]
            if len(tokens) < self.max_seq_len:
                tokens = tokens + [self.vocab['<PAD>']] * (self.max_seq_len - len(tokens))
            else:
                tokens = tokens[:self.max_seq_len]
            tokenized.append(tokens)
        return torch.tensor(tokenized, dtype=torch.long)

    def __len__(self):
        return len(self.icd_labels)

    def __getitem__(self, idx):
        return self.num_features[idx], self.text_features[idx], self.icd_labels[idx], self.chexpert_labels[idx]


class EFNet(nn.Module):
    def __init__(self, n_features, seq_length, vocab_size, embedding_dim=64, n_classes=8):
        super(EFNet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.prelu_1 = nn.PReLU()
        self.embedding_l1 = nn.Linear(seq_length * embedding_dim, 32)
        self.prelu_2 = nn.PReLU()
        self.embedding_to_add = nn.Linear(32, 16)
        self.prelu_3 = nn.PReLU()

        self.numerical_to_match = nn.Linear(n_features, 16)
        self.prelu_4 = nn.PReLU()

        self.prelu_5 = nn.PReLU()
        self.classify = nn.Linear(16, n_classes)

    def forward(self, n, c):
        c_out1 = self.embedding(c)
        c_out1 = torch.flatten(c_out1, start_dim=1)
        c_out1 = self.prelu_1(c_out1)
        c_out1 = self.embedding_l1(c_out1)
        c_out1 = self.prelu_2(c_out1)
        c_out1 = self.embedding_to_add(c_out1)
        c_out1 = self.prelu_3(c_out1)

        n_out1 = self.numerical_to_match(n)
        n_out1 = self.prelu_4(n_out1)

        out = n_out1 + c_out1
        out = self.prelu_5(out)
        return self.classify(out)


# ==========================================
# 2. FUNCȚIA DE GENERARE GRAFICE (MATRICEA DE CONFUZIE)
# ==========================================
def plot_13_confusion_matrices(y_true, y_pred, class_names):
    sns.set_theme(style="white")
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    axes = axes.flatten()

    for i, class_name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        annot_text = np.array([
            [f"TN\n{tn}", f"FP\n{fp}"],
            [f"FN\n{fn}", f"TP\n{tp}"]
        ])

        sns.heatmap(cm, annot=annot_text, fmt="", cmap="Blues", cbar=False,
                    xticklabels=['Negativ', 'Pozitiv'],
                    yticklabels=['Negativ', 'Pozitiv'],
                    ax=axes[i], annot_kws={"size": 14, "weight": "bold"})

        axes[i].set_title(class_name, fontsize=14, fontweight='bold', pad=10)
        axes[i].set_xlabel('Predicție Rețea')
        axes[i].set_ylabel('Adevărul (Ground Truth)')

    for j in range(13, 16):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("Matrici_Confuzie_13_Patologii.png", dpi=300, bbox_inches='tight')
    plt.show()


# ==========================================
# 3. PIPELINE DE EVALUARE
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluare pe device: {device}")

    # --- RECONSTRUIREA MEDIULUI EXACT CA LA ANTRENARE ---
    csv_path = r"/dataset_multimodal_final.csv"
    df = pd.read_csv(csv_path)

    chexpert_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                     'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                     'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
                     'Pneumonia', 'Pneumothorax', 'Support Devices']
    icd_cols = [c for c in df.columns if c.startswith('ICD_')]
    text_col = 'chiefcomplaint'
    num_cols = [c for c in df.columns if
                c not in chexpert_cols + icd_cols + ['No Finding', 'subject_id', 'stay_id', 'study_id',
                                                     'chiefcomplaint', 'gender', 'race', 'arrival_transport']]

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_dataset = MultimodalMIMICDataset(train_df, text_col, num_cols, icd_cols, chexpert_cols)
    test_dataset = MultimodalMIMICDataset(test_df, text_col, num_cols, icd_cols, chexpert_cols,
                                          vocab=train_dataset.vocab, max_seq_len=train_dataset.max_seq_len,
                                          scaler=train_dataset.scaler, imputer=train_dataset.imputer)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # --- ÎNCĂRCAREA MODELULUI ---
    model_path = 'best_model_Phase2_CheXpert.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nu găsesc fișierul {model_path}! Asigură-te că scriptul e în același folder.")

    safe_vocab_size = max(train_dataset.vocab.values()) + 1

    model = EFNet(n_features=len(num_cols),
                  seq_length=train_dataset.max_seq_len,
                  vocab_size=safe_vocab_size,
                  n_classes=len(chexpert_cols)).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- RULAREA PREDICȚIILOR ---
    print("\nRulare predicții pe setul de test...")
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for num_feats, text_feats, _, chexpert_labels in test_loader:
            num_feats, text_feats = num_feats.to(device), text_feats.to(device)
            targets = chexpert_labels.to(device)

            outputs = model(num_feats, text_feats)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # ==========================================
    # CALCULUL METRICILOR PER CLASĂ ȘI MACRO
    # ==========================================
    print("\n" + "=" * 60)
    print(f"{'PATOLOGIE':<30} | {'AUC':<6} | {'PREC':<6} | {'REC':<6} | {'F1':<6}")
    print("=" * 60)

    # Calculăm valorile pentru FIECARE clasă în parte (folosind average=None)
    # Poate apărea un mic warning de la sklearn dacă o clasă nu are exemple pozitive,
    # de aceea folosim zero_division=0.
    auc_per_class = roc_auc_score(all_targets, all_probs, average=None)
    prec_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
    rec_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)

    # Afișăm frumos tabelat fiecare patologie
    for i, patologie in enumerate(chexpert_cols):
        print(
            f"{patologie:<30} | {auc_per_class[i]:.4f} | {prec_per_class[i]:.4f} | {rec_per_class[i]:.4f} | {f1_per_class[i]:.4f}")

    print("=" * 60)

    # Calculăm mediile globale (Macro Average)
    auc_macro = roc_auc_score(all_targets, all_probs, average='macro')
    prec_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f"{'MEDIE GLOBALĂ (MACRO)':<30} | {auc_macro:.4f} | {prec_macro:.4f} | {rec_macro:.4f} | {f1_macro:.4f}")
    print("=" * 60)

    # Generăm și salvăm graficul
    print("\nGenerare grafic cu cele 13 Matrici de Confuzie...")
    plot_13_confusion_matrices(all_targets, all_preds, chexpert_cols)
    print("Graficul a fost salvat ca 'Matrici_Confuzie_13_Patologii.png' în folderul curent!")



## SCRIPT EVALUARE OVERSAMPLING+POSWEIGHT
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter
# import os
#
# # Setări reproductibilitate
# torch.manual_seed(42)
# np.random.seed(42)
#
#
# # ==========================================
# # 1. CLASELE NECESARE
# # ==========================================
# class MultimodalMIMICDataset(Dataset):
#     def __init__(self, df, text_col, num_cols, icd_cols, chexpert_cols, vocab=None, max_seq_len=None, scaler=None,
#                  imputer=None):
#         self.icd_labels = torch.tensor(df[icd_cols].fillna(0).values, dtype=torch.float32)
#         self.chexpert_labels = torch.tensor(df[chexpert_cols].fillna(0).values, dtype=torch.float32)
#
#         if imputer is None:
#             self.imputer = SimpleImputer(strategy='mean')
#             num_data_imputed = self.imputer.fit_transform(df[num_cols])
#         else:
#             self.imputer = imputer
#             num_data_imputed = self.imputer.transform(df[num_cols])
#
#         if scaler is None:
#             self.scaler = StandardScaler()
#             num_data = self.scaler.fit_transform(num_data_imputed)
#         else:
#             self.scaler = scaler
#             num_data = self.scaler.transform(num_data_imputed)
#
#         num_data = np.nan_to_num(num_data, nan=0.0)
#         self.num_features = torch.tensor(num_data, dtype=torch.float32)
#
#         self.text_data = df[text_col].fillna("").astype(str).tolist()
#
#         if vocab is None:
#             self.vocab, self.max_seq_len = self._build_vocab(self.text_data)
#         else:
#             self.vocab = vocab
#             self.max_seq_len = max_seq_len
#
#         self.text_features = self._tokenize_and_pad(self.text_data)
#
#     def _build_vocab(self, texts):
#         words = " ".join(texts).split()
#         counter = Counter(words)
#         vocab = {'<PAD>': 0, '<UNK>': 1}
#         idx = 2
#         for word, count in counter.items():
#             if count > 1:
#                 vocab[word] = idx
#                 idx += 1
#         max_len = max(len(text.split()) for text in texts)
#         max_len = min(max_len, 20)
#         return vocab, max_len
#
#     def _tokenize_and_pad(self, texts):
#         tokenized = []
#         for text in texts:
#             tokens = [self.vocab.get(w, self.vocab['<UNK>']) for w in text.split()]
#             if len(tokens) < self.max_seq_len:
#                 tokens = tokens + [self.vocab['<PAD>']] * (self.max_seq_len - len(tokens))
#             else:
#                 tokens = tokens[:self.max_seq_len]
#             tokenized.append(tokens)
#         return torch.tensor(tokenized, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.icd_labels)
#
#     def __getitem__(self, idx):
#         return self.num_features[idx], self.text_features[idx], self.icd_labels[idx], self.chexpert_labels[idx]
#
#
# class EFNet(nn.Module):
#     def __init__(self, n_features, seq_length, vocab_size, embedding_dim=64, n_classes=8):
#         super(EFNet, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
#         self.prelu_1 = nn.PReLU()
#         self.embedding_l1 = nn.Linear(seq_length * embedding_dim, 32)
#         self.prelu_2 = nn.PReLU()
#         self.embedding_to_add = nn.Linear(32, 16)
#         self.prelu_3 = nn.PReLU()
#
#         self.numerical_to_match = nn.Linear(n_features, 16)
#         self.prelu_4 = nn.PReLU()
#
#         self.prelu_5 = nn.PReLU()
#         self.classify = nn.Linear(16, n_classes)
#
#     def forward(self, n, c):
#         c_out1 = self.embedding(c)
#         c_out1 = torch.flatten(c_out1, start_dim=1)
#         c_out1 = self.prelu_1(c_out1)
#         c_out1 = self.embedding_l1(c_out1)
#         c_out1 = self.prelu_2(c_out1)
#         c_out1 = self.embedding_to_add(c_out1)
#         c_out1 = self.prelu_3(c_out1)
#
#         n_out1 = self.numerical_to_match(n)
#         n_out1 = self.prelu_4(n_out1)
#
#         out = n_out1 + c_out1
#         out = self.prelu_5(out)
#         return self.classify(out)
#
#
# # ==========================================
# # 2. FUNCȚII AUXILIARE (Oversampling & Plot)
# # ==========================================
# def oversample_rare_classes(df, target_cols, threshold_ratio=0.1, duplications=2):
#     new_dfs = [df]
#     n_total = len(df)
#     for col in target_cols:
#         pos_count = df[col].sum()
#         if 0 < pos_count < (threshold_ratio * n_total):
#             minority_df = df[df[col] == 1]
#             for _ in range(duplications - 1):
#                 new_dfs.append(minority_df)
#     oversampled_df = pd.concat(new_dfs, ignore_index=True)
#     return oversampled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
#
#
# def plot_13_confusion_matrices(y_true, y_pred, class_names):
#     sns.set_theme(style="white")
#     fig, axes = plt.subplots(4, 4, figsize=(18, 16))
#     axes = axes.flatten()
#
#     for i, class_name in enumerate(class_names):
#         cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
#         tn, fp, fn, tp = cm.ravel()
#
#         annot_text = np.array([
#             [f"TN\n{tn}", f"FP\n{fp}"],
#             [f"FN\n{fn}", f"TP\n{tp}"]
#         ])
#
#         sns.heatmap(cm, annot=annot_text, fmt="", cmap="Blues", cbar=False,
#                     xticklabels=['Negativ', 'Pozitiv'],
#                     yticklabels=['Negativ', 'Pozitiv'],
#                     ax=axes[i], annot_kws={"size": 14, "weight": "bold"})
#
#         axes[i].set_title(class_name, fontsize=14, fontweight='bold', pad=10)
#         axes[i].set_xlabel('Predicție Rețea')
#         axes[i].set_ylabel('Adevărul (Ground Truth)')
#
#     for j in range(13, 16):
#         fig.delaxes(axes[j])
#
#     plt.tight_layout()
#     plt.savefig("Matrici_Confuzie_13_Patologii.png", dpi=300, bbox_inches='tight')
#     plt.show()
#
#
# # ==========================================
# # 3. PIPELINE DE EVALUARE
# # ==========================================
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Evaluare pe device: {device}")
#
#     # --- RECONSTRUIREA MEDIULUI EXACT CA LA ANTRENARE ---
#     csv_path = r"C:\Users\2D\PycharmProjects\licenta_pt\dataset_multimodal_final.csv"
#     df = pd.read_csv(csv_path)
#
#     chexpert_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
#                      'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
#                      'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
#                      'Pneumonia', 'Pneumothorax', 'Support Devices']
#     icd_cols = [c for c in df.columns if c.startswith('ICD_')]
#     text_col = 'chiefcomplaint'
#     num_cols = [c for c in df.columns if
#                 c not in chexpert_cols + icd_cols + ['No Finding', 'subject_id', 'stay_id', 'study_id',
#                                                      'chiefcomplaint', 'gender', 'race', 'arrival_transport']]
#
#     train_df_original, temp_df = train_test_split(df, test_size=0.3, random_state=42)
#     _, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
#
#     # APLICĂM OVERSAMPLING AICI PENTRU A GENERA ACELAȘI VOCABULAR DE 2018 CUVINTE
#     train_df = oversample_rare_classes(train_df_original, chexpert_cols, threshold_ratio=0.1, duplications=2)
#
#     train_dataset = MultimodalMIMICDataset(train_df, text_col, num_cols, icd_cols, chexpert_cols)
#     test_dataset = MultimodalMIMICDataset(test_df, text_col, num_cols, icd_cols, chexpert_cols,
#                                           vocab=train_dataset.vocab, max_seq_len=train_dataset.max_seq_len,
#                                           scaler=train_dataset.scaler, imputer=train_dataset.imputer)
#
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
#
#     # --- ÎNCĂRCAREA MODELULUI ---
#     model_path = 'best_model_Phase2_CheXpert.pth'
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Nu găsesc fișierul {model_path}!")
#
#     safe_vocab_size = max(train_dataset.vocab.values()) + 1
#
#     model = EFNet(n_features=len(num_cols),
#                   seq_length=train_dataset.max_seq_len,
#                   vocab_size=safe_vocab_size,
#                   n_classes=len(chexpert_cols)).to(device)
#
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#
#     # --- RULAREA PREDICȚIILOR ---
#     print("\nRulare predicții pe setul de test...")
#     all_preds = []
#     all_probs = []
#     all_targets = []
#
#     with torch.no_grad():
#         for num_feats, text_feats, _, chexpert_labels in test_loader:
#             num_feats, text_feats = num_feats.to(device), text_feats.to(device)
#             targets = chexpert_labels.to(device)
#
#             outputs = model(num_feats, text_feats)
#             probs = torch.sigmoid(outputs)
#             preds = (probs > 0.5).float()
#
#             all_probs.append(probs.cpu().numpy())
#             all_preds.append(preds.cpu().numpy())
#             all_targets.append(targets.cpu().numpy())
#
#     all_probs = np.vstack(all_probs)
#     all_preds = np.vstack(all_preds)
#     all_targets = np.vstack(all_targets)
#
#     # ==========================================
#     # CALCULUL METRICILOR
#     # ==========================================
#     print("\n" + "=" * 60)
#     print(f"{'PATOLOGIE':<30} | {'AUC':<6} | {'PREC':<6} | {'REC':<6} | {'F1':<6}")
#     print("=" * 60)
#
#     auc_per_class = roc_auc_score(all_targets, all_probs, average=None)
#     prec_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
#     rec_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)
#     f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)
#
#     for i, patologie in enumerate(chexpert_cols):
#         print(
#             f"{patologie:<30} | {auc_per_class[i]:.4f} | {prec_per_class[i]:.4f} | {rec_per_class[i]:.4f} | {f1_per_class[i]:.4f}")
#
#     print("=" * 60)
#
#     auc_macro = roc_auc_score(all_targets, all_probs, average='macro')
#     prec_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
#     rec_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
#     f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
#
#     print(f"{'MEDIE GLOBALĂ (MACRO)':<30} | {auc_macro:.4f} | {prec_macro:.4f} | {rec_macro:.4f} | {f1_macro:.4f}")
#     print("=" * 60)
#
#     print("\nGenerare grafic cu cele 13 Matrici de Confuzie...")
#     plot_13_confusion_matrices(all_targets, all_preds, chexpert_cols)
#     print("Graficul a fost salvat ca 'Matrici_Confuzie_13_Patologii.png' în folderul curent!")