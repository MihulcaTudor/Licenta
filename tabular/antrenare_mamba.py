import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from mamba_ssm import Mamba
import math
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURARE CONFORM EHRMAMBA
# ============================================================
CSV_PATH = "ehrmamba_ed_dataset_2.csv"
MAX_SEQ_LEN = 256
BATCH_SIZE = 16 # Scăzut de la 64 pentru a preveni OOM pe 32 straturi, ajustează în funcție de GPU
EPOCHS = 20      # 6 Conform lucrării pentru finetuning
MAX_LR = 5e-5   # Conform tabelului 4 din lucrare

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. DATASET ȘI VOCABULAR
# ============================================================
def build_vocab(df):
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[VS]", "[VE]", "[REG]"]
    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    idx = len(vocab)
    for seq in df['concept_seq']:
        for token in str(seq).split(','):
            token = token.strip()
            if token not in vocab:
                vocab[token] = idx
                idx += 1
    return vocab


class EHRDataset(Dataset):
    def __init__(self, df, vocab, max_len=256):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        concepts = str(row['concept_seq']).split(',')
        times = [float(x) / 60.0 for x in str(row['time_seq']).split(',')]

        types = [int(float(x)) + 1 for x in str(row['type_seq']).split(',')]
        ages = [min(118, int(float(x))) + 1 for x in str(row['age_seq']).split(',')]
        segments = [int(float(x)) + 1 for x in str(row['segment_seq']).split(',')]
        visit_orders = [int(float(x)) + 1 for x in str(row['visit_order_seq']).split(',')]

        concepts = concepts[:self.max_len]
        times = times[:self.max_len]
        types = types[:self.max_len]
        ages = ages[:self.max_len]
        segments = segments[:self.max_len]
        visit_orders = visit_orders[:self.max_len]

        real_seq_len = len(concepts)
        concept_ids = [self.vocab.get(c, self.vocab['[UNK]']) for c in concepts]

        pad_len = self.max_len - real_seq_len
        concept_ids += [0] * pad_len
        times += [0.0] * pad_len
        types += [0] * pad_len
        ages += [0] * pad_len
        segments += [0] * pad_len
        visit_orders += [0] * pad_len

        labels = row[LABEL_COLS].values.astype(np.float32)

        return {
            'concept_ids': torch.tensor(concept_ids, dtype=torch.long),
            'time_vals': torch.tensor(times, dtype=torch.float),
            'type_ids': torch.tensor(types, dtype=torch.long),
            'age_ids': torch.tensor(ages, dtype=torch.long),
            'segment_ids': torch.tensor(segments, dtype=torch.long),
            'visit_order_ids': torch.tensor(visit_orders, dtype=torch.long),
            'seq_lens': torch.tensor(real_seq_len, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


# ============================================================
# 3. ARHITECTURA EHRMAMBA
# ============================================================
class Time2Vec(nn.Module):
    def __init__(self, time_emb_size=32):
        super().__init__()
        # Inițializare cu valori foarte mici pentru a preveni explozia (NaN)
        self.w0 = nn.Parameter(torch.randn(1, 1) * 0.01)
        self.b0 = nn.Parameter(torch.randn(1, 1) * 0.01)
        self.w = nn.Parameter(torch.randn(1, time_emb_size - 1) * 0.01)
        self.b = nn.Parameter(torch.randn(1, time_emb_size - 1) * 0.01)

    def forward(self, tau):
        tau = tau.unsqueeze(-1)
        v1 = tau * self.w0 + self.b0
        v2 = torch.sin(tau * self.w + self.b)
        return torch.cat([v1, v2], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.d = d
        self.weight = nn.Parameter(torch.ones(d))
        self.register_parameter('bias', nn.Parameter(torch.zeros(d)) if bias else None)

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        d_x = self.d
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias is not None:
            return self.weight * x_normed + self.bias
        return self.weight * x_normed


class EHRMambaModel(nn.Module):
    def __init__(self, vocab_size, num_classes=13, pos_rates=None):
        super().__init__()
        # Hiperparametri conform Tabel 4 (EHRMAMBA)
        d_model = 768
        d_state = 16
        d_conv = 4
        expand = 2
        n_layers = 12   # Notă: Lucrarea folosește 32, dar pentru max_len=256 de la zero, recomand 12 ca punct de plecare. Crește la 32 dacă ai VRAM (A100).

        self.concept_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(10, d_model, padding_idx=0)
        self.age_emb = nn.Embedding(125, d_model, padding_idx=0)
        self.segment_emb = nn.Embedding(10, d_model, padding_idx=0)
        self.visit_order_emb = nn.Embedding(100, d_model, padding_idx=0)

        self.time2vec = Time2Vec(time_emb_size=32)
        self.time_proj = nn.Linear(32, d_model)

        # Mamba Blocks
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])

        # Normalizare și Clasificare
        self.norm_f = RMSNorm(d_model)
        self.dropout = nn.Dropout(0.1) # Conform lucrării
        self.classifier = nn.Linear(d_model, num_classes)

        # Initializarea Karpathy pentru antrenare de la zero
        if pos_rates is not None:
            initial_bias = np.log(pos_rates / (1 - pos_rates + 1e-5))
            self.classifier.bias.data = torch.tensor(initial_bias, dtype=torch.float)
        self.emb_norm = RMSNorm(d_model)  # Stabilizator pentru suma de embeddings

    def forward(self, concept_ids, type_ids, age_ids, segment_ids, visit_order_ids, time_vals, seq_lens):
        e_c = self.concept_emb(concept_ids)
        e_ty = self.type_emb(type_ids)
        e_a = self.age_emb(age_ids)
        e_s = self.segment_emb(segment_ids)
        e_v = self.visit_order_emb(visit_order_ids)
        e_ti = self.time_proj(self.time2vec(time_vals))

        x = e_c + e_ty + e_a + e_s + e_v + e_ti
        x = self.emb_norm(x)  # NORMALIZĂM SUMA înainte de a intra în Mamba!

        for layer in self.layers:
            x = layer(x)


        x = self.norm_f(x)

        batch_size = x.size(0)
        last_token_indices = seq_lens - 1
        h_last = x[torch.arange(batch_size, device=x.device), last_token_indices, :]

        # Sigmoid este aplicat automat mai târziu în BCEWithLogitsLoss
        logits = self.classifier(self.dropout(h_last))
        return logits


# ============================================================
# 4. SCRIPTUL DE ANTRENARE ROBUST
# ============================================================
def get_pos_weights(df):
    labels = df[LABEL_COLS].values
    num_positives = labels.sum(axis=0)
    num_negatives = len(df) - num_positives

    pos_weights = num_negatives / (num_positives + 10.0)
    pos_weights = np.clip(pos_weights, 1.0, 15.0)
    return torch.tensor(pos_weights, dtype=torch.float).to(device)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Scheduler liniar conform specificațiilor EHRMAMBA:
    Crește liniar în primele 10% (warmup), apoi scade liniar în restul de 90%.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model():
    df = pd.read_csv(CSV_PATH)
    vocab = build_vocab(df)

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_loader = DataLoader(EHRDataset(train_df, vocab, MAX_SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EHRDataset(val_df, vocab, MAX_SEQ_LEN), batch_size=BATCH_SIZE)

    pos_rates = train_df[LABEL_COLS].mean().values
    print(f"Rata de cazuri pozitive per clasă: {np.round(pos_rates, 3)}")

    model = EHRMambaModel(vocab_size=len(vocab), num_classes=13, pos_rates=pos_rates).to(device)

    # Optimizator AdamW (Decoupled weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1) # 10% warmup conform lucrării
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    pos_weights = get_pos_weights(train_df)
    print(f"Ponderi clase: {pos_weights.cpu().numpy().round(2)}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)



    best_val_loss = float('inf')

    scaler = torch.cuda.amp.GradScaler()  # Adaugă asta!
    print("\n--- START ANTRENARE EHRMAMBA ---")

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0

        train_loop = tqdm(train_loader, desc=f"Epoca {epoch + 1}/{EPOCHS} [Train]", leave=False)
        for batch in train_loop:
            optimizer.zero_grad()

            c_ids = batch['concept_ids'].to(device)
            ty_ids = batch['type_ids'].to(device)
            a_ids = batch['age_ids'].to(device)
            s_ids = batch['segment_ids'].to(device)
            v_ids = batch['visit_order_ids'].to(device)
            tm_vals = batch['time_vals'].to(device)
            s_lens = batch['seq_lens'].to(device)
            labels = batch['labels'].to(device)

            # Rulăm forward în contextul autocast pentru Mixed Precision
            with torch.cuda.amp.autocast():
                logits = model(c_ids, ty_ids, a_ids, s_ids, v_ids, tm_vals, s_lens)
                loss = criterion(logits, labels)

            # Backward prin Scaler
            scaler.scale(loss).backward()

            # Înainte de clipping, trebuie să facem unscale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Pasul de optimizare prin Scaler
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            total_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        model.eval()
        val_loss = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in val_loader:
                c_ids = batch['concept_ids'].to(device)
                ty_ids = batch['type_ids'].to(device)
                a_ids = batch['age_ids'].to(device)
                s_ids = batch['segment_ids'].to(device)
                v_ids = batch['visit_order_ids'].to(device)
                tm_vals = batch['time_vals'].to(device)
                s_lens = batch['seq_lens'].to(device)
                labels = batch['labels'].to(device)

                logits = model(c_ids, ty_ids, a_ids, s_ids, v_ids, tm_vals, s_lens)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(logits)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)

        try:
            valid_cols = all_labels.sum(axis=0) > 0
            if valid_cols.sum() > 0:
                auc = roc_auc_score(all_labels[:, valid_cols], all_preds[:, valid_cols], average='macro')
            else:
                auc = 0.0
        except Exception:
            auc = 0.0

        f1 = f1_score(all_labels, all_preds > 0.3, average='macro', zero_division=0)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time

        print(
            f"\nEpoca [{epoch + 1}/{EPOCHS}] | Timp: {epoch_time:.1f}s | LR final epocă: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "ehrmamba_best.pth")
            print(f"  --> Model salvat! (Val Loss a scăzut)")


if __name__ == "__main__":
    train_model()