# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from collections import Counter
# import os
#
# # Setări reproductibilitate
# torch.manual_seed(42)
# np.random.seed(42)
#
#
# # ==========================================
# # 1. CLASA DATASET (cu Imputer)
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
# # ==========================================
# # 2. ARHITECTURA EF-NET
# # ==========================================
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
# # 3. FUNCȚII UTILS:  doar Pos Weight
# # ==========================================
# # def oversample_rare_classes(df, target_cols, threshold_ratio=0.1, duplications=2):
# # #     """
# # #     Dublează pacienții care au boli rare (frecvență sub un anumit prag).
# # #     """
# # #     new_dfs = [df]  # Păstrăm tot setul original
# # #     n_total = len(df)
# # #
# # #     for col in target_cols:
# # #         pos_count = df[col].sum()
# # #         # Dacă boala e prezentă la mai puțin de 10% (threshold_ratio) din pacienți
# # #         if 0 < pos_count < (threshold_ratio * n_total):
# # #             minority_df = df[df[col] == 1]
# # #             for _ in range(duplications - 1):  # Adăugăm copii suplimentare
# # #                 new_dfs.append(minority_df)
# # #
# # #     # Unim și amestecăm bine datele
# # #     oversampled_df = pd.concat(new_dfs, ignore_index=True)
# # #     return oversampled_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
#
#
# def get_dynamic_pos_weights(df, cols, device):
#     """
#     Calculează pos_weight pentru BCEWithLogitsLoss.
#     Oprește valorile la un prag maxim (ex: 20)
#     """
#     pos_counts = df[cols].sum().values
#     neg_counts = len(df) - pos_counts
#
#     # Evităm împărțirea la zero
#     weights = neg_counts / (pos_counts + 1e-5)
#
#     # Tăiem greutățile extreme
#     weights = np.clip(weights, 1.0, 20.0)
#
#     return torch.tensor(weights, dtype=torch.float32).to(device)
#
#
# # ==========================================
# # 4. FUNCȚIA DE ANTRENARE
# # ==========================================
# def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, phase_name, target_idx):
#     best_val_loss = float('inf')
#
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#
#         for num_feats, text_feats, icd_labels, chexpert_labels in train_loader:
#             num_feats, text_feats = num_feats.to(device), text_feats.to(device)
#             targets = icd_labels.to(device) if target_idx == 0 else chexpert_labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(num_feats, text_feats)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for num_feats, text_feats, icd_labels, chexpert_labels in val_loader:
#                 num_feats, text_feats = num_feats.to(device), text_feats.to(device)
#                 targets = icd_labels.to(device) if target_idx == 0 else chexpert_labels.to(device)
#
#                 outputs = model(num_feats, text_feats)
#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item()
#
#         train_loss = running_loss / len(train_loader)
#         val_loss = val_loss / len(val_loader)
#
#         print(f"[{phase_name}] Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), f'best_model_{phase_name}.pth')
#
#
# # ==========================================
# # 5. EXECUȚIA PIPELINE-ULUI
# # ==========================================
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Antrenare pe device: {device}")
#
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
#     # Split standard (fără nicio modificare asupra datelor de antrenament)
#     train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
#     val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
#
#     print(f"\nDimensiune Train: {train_df.shape[0]}")
#
#     #     # 2. APLICĂM OVERSAMPLING DOAR PE TRAIN
#     #     print(f"\nDimensiune Train Original: {train_df_original.shape[0]}")
#     #     train_df = oversample_rare_classes(train_df_original, chexpert_cols, threshold_ratio=0.1, duplications=2)
#     #     print(f"Dimensiune Train DUPĂ Oversampling (Boli rare dublate): {train_df.shape[0]}")
#
#     # Creăm Dataset-urile (strict din datele originale)
#     train_dataset = MultimodalMIMICDataset(train_df, text_col, num_cols, icd_cols, chexpert_cols)
#     val_dataset = MultimodalMIMICDataset(val_df, text_col, num_cols, icd_cols, chexpert_cols,
#                                          vocab=train_dataset.vocab, max_seq_len=train_dataset.max_seq_len,
#                                          scaler=train_dataset.scaler, imputer=train_dataset.imputer)
#
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
#
#     # ==========================================
#     # FAZA 1 (Pre-training pe coduri ICD)
#     # ==========================================
#     print("\n--- INIȚIERE FAZA 1: Pre-training pe Coduri ICD-10 ---")
#     safe_vocab_size = max(train_dataset.vocab.values()) + 1
#     model = EFNet(n_features=len(num_cols), seq_length=train_dataset.max_seq_len,
#                   vocab_size=safe_vocab_size, n_classes=len(icd_cols)).to(device)
#
#     # Greutăți dinamice pentru Faza 1
#     pos_weight_icd = get_dynamic_pos_weights(train_df, icd_cols, device)
#     criterion_icd = nn.BCEWithLogitsLoss(pos_weight=pos_weight_icd)
#
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     train_model(model, train_loader, val_loader, criterion_icd, optimizer, device, epochs=50, phase_name="Phase1_ICD",
#                 target_idx=0)
#
#     # ==========================================
#     # FAZA 2 (Fine-tuning pe CheXpert)
#     # ==========================================
#     print("\n--- INIȚIERE FAZA 2: Fine-tuning pe 13 patologii CheXpert ---")
#     model.load_state_dict(torch.load('best_model_Phase1_ICD.pth'))
#     model.classify = nn.Linear(16, len(chexpert_cols)).to(device)
#
#     # Greutăți dinamice pentru Faza 2
#     pos_weight_chexpert = get_dynamic_pos_weights(train_df, chexpert_cols, device)
#     print(f"Ponderi aplicate funcției de cost: \n{pos_weight_chexpert.cpu().numpy()}")
#
#     criterion_chexpert = nn.BCEWithLogitsLoss(pos_weight=pos_weight_chexpert)
#     optimizer_ft = optim.Adam(model.parameters(), lr=0.0005)
#
#     train_model(model, train_loader, val_loader, criterion_chexpert, optimizer_ft, device, epochs=50,
#                 phase_name="Phase2_CheXpert", target_idx=1)
#
#     print("\nAntrenament complet! Modelele au fost salvate.")
#
#

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import Counter
import os

# Setări reproductibilitate
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# 1. CLASA DATASET (cu Imputer)
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


# ==========================================
# 2. ARHITECTURA EF-NET
# ==========================================
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
# 3. FUNCȚII UTILS: Pos Weight
# ==========================================
def get_dynamic_pos_weights(df, cols, device):
    """
    Calculează pos_weight pentru BCEWithLogitsLoss.
    Oprește valorile la un prag maxim (ex: 20) pentru a nu destabiliza modelul.
    """
    pos_counts = df[cols].sum().values
    neg_counts = len(df) - pos_counts

    # Evităm împărțirea la zero
    weights = neg_counts / (pos_counts + 1e-5)

    # Tăiem greutățile extreme
    weights = np.clip(weights, 1.0, 20.0)

    return torch.tensor(weights, dtype=torch.float32).to(device)


# ==========================================
# 4. FUNCȚIA DE ANTRENARE
# ==========================================
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, phase_name, target_idx):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for num_feats, text_feats, icd_labels, chexpert_labels in train_loader:
            num_feats, text_feats = num_feats.to(device), text_feats.to(device)
            targets = icd_labels.to(device) if target_idx == 0 else chexpert_labels.to(device)

            optimizer.zero_grad()
            outputs = model(num_feats, text_feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for num_feats, text_feats, icd_labels, chexpert_labels in val_loader:
                num_feats, text_feats = num_feats.to(device), text_feats.to(device)
                targets = icd_labels.to(device) if target_idx == 0 else chexpert_labels.to(device)

                outputs = model(num_feats, text_feats)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)

        print(f"[{phase_name}] Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_efnet_{phase_name}.pth')


# ==========================================
# 5. EXECUȚIA PIPELINE-ULUI (ANTRENARE DIRECTĂ)
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Antrenare pe device: {device}")

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

    # Split standard (fără nicio modificare asupra datelor de antrenament)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"\nDimensiune Train: {train_df.shape[0]}")

    # Creăm Dataset-urile
    train_dataset = MultimodalMIMICDataset(train_df, text_col, num_cols, icd_cols, chexpert_cols)
    val_dataset = MultimodalMIMICDataset(val_df, text_col, num_cols, icd_cols, chexpert_cols,
                                         vocab=train_dataset.vocab, max_seq_len=train_dataset.max_seq_len,
                                         scaler=train_dataset.scaler, imputer=train_dataset.imputer)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # ==========================================
    # ANTRENARE DIRECTĂ PE CHEXPERT (BASELINE)
    # ==========================================
    print("\n--- INIȚIERE DIRECTĂ: Antrenament de la zero pe 13 patologii CheXpert ---")
    safe_vocab_size = max(train_dataset.vocab.values()) + 1

    # Inițializăm modelul direct cu 13 clase de ieșire
    model = EFNet(n_features=len(num_cols), seq_length=train_dataset.max_seq_len,
                  vocab_size=safe_vocab_size, n_classes=len(chexpert_cols)).to(device)

    # Greutăți dinamice pentru CheXpert
    pos_weight_chexpert = get_dynamic_pos_weights(train_df, chexpert_cols, device)
    print(f"Ponderi aplicate funcției de cost: \n{pos_weight_chexpert.cpu().numpy()}")

    criterion_chexpert = nn.BCEWithLogitsLoss(pos_weight=pos_weight_chexpert)

    # Folosim LR normal (0.001) pt că modelul nu are greutăți pre-antrenate
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Parametrul target_idx=1 indică loader-ului să folosească etichetele CheXpert
    train_model(model, train_loader, val_loader, criterion_chexpert, optimizer, device, epochs=50,
                phase_name="Direct_CheXpert", target_idx=1)

    print("\nAntrenament complet! Modelul a fost salvat sub numele 'best_model_Direct_CheXpert.pth'.")