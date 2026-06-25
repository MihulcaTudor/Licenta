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
from transformers import BertConfig, get_cosine_schedule_with_warmup
from transformers.models.bert.modeling_bert import BertEncoder
import warnings
import os

# Ignorăm warning-urile sklearn pentru clasele rare
warnings.filterwarnings('ignore')

# Forțăm compatibilitatea cu generațiile anterioare pentru RTX 5070
os.environ['PyTorch_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'

# ============================================================
# 1. CONFIGURARE
# ============================================================
CSV_PATH = "ehrmamba_ed_dataset_2.csv"  # Asigură-te că numele fișierului este corect
MAX_SEQ_LEN = 256
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 5e-5

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 2. DATASET ȘI VOCABULAR
# ============================================================
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
        times = [float(x) for x in str(row['time_seq']).split(',')]
        types = [int(x) for x in str(row['type_seq']).split(',')]
        positions = [int(x) for x in str(row['position_seq']).split(',')]

        concepts = concepts[:self.max_len]
        times = times[:self.max_len]
        types = types[:self.max_len]
        positions = positions[:self.max_len]

        concept_ids = [self.vocab.get(c, self.vocab['[UNK]']) for c in concepts]
        pad_len = self.max_len - len(concept_ids)

        concept_ids += [0] * pad_len
        times += [0.0] * pad_len
        types += [0] * pad_len
        positions += [0] * pad_len

        # Mască de atenție (1 real, 0 padding)
        attention_mask = [1] * (self.max_len - pad_len) + [0] * pad_len
        labels = row[LABEL_COLS].values.astype(np.float32)

        return {
            'concept_ids': torch.tensor(concept_ids, dtype=torch.long),
            'type_ids': torch.tensor(types, dtype=torch.long),
            'time_vals': torch.tensor(times, dtype=torch.float),
            'pos_ids': torch.tensor(positions, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float)
        }


def build_vocab(df):
    unique_tokens = set(["[PAD]", "[UNK]", "[CLS]", "[VS]", "[VE]", "[REG]"])
    for seq in df['concept_seq']:
        for token in str(seq).split(','):
            unique_tokens.add(token.strip())
    return {token: idx for idx, token in enumerate(sorted(unique_tokens))}


# ============================================================
# 3. ARHITECTURA CEHR-BERT
# ============================================================
class Time2Vec(nn.Module):
    def __init__(self, time_emb_size=32):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.b0 = nn.Parameter(torch.randn(1, 1))
        self.w = nn.Parameter(torch.randn(1, time_emb_size - 1))
        self.b = nn.Parameter(torch.randn(1, time_emb_size - 1))

    def forward(self, tau):
        tau = tau.unsqueeze(-1)
        v1 = tau * self.w0 + self.b0
        v2 = torch.sin(tau * self.w + self.b)
        return torch.cat([v1, v2], dim=-1)


class CEHRBERTModel(nn.Module):
    def __init__(self, vocab_size, num_classes=13):
        super().__init__()
        d_model = 768

        # 3.1. Embedding-uri paralele
        self.concept_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # MĂRIT DE LA 4 LA 10 pentru a acoperi id-urile din type_seq (ex: id-ul 4)
        self.type_emb = nn.Embedding(10, d_model, padding_idx=0)

        # MĂRIT DE LA MAX_SEQ_LEN (256) la 512 pentru siguranță
        self.pos_emb = nn.Embedding(512, d_model)

        self.time2vec = Time2Vec(time_emb_size=32)
        self.time_proj = nn.Linear(32, d_model)


        self.LayerNorm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(0.1)

        config = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            attn_implementation = "eager"
        )
        self.encoder = BertEncoder(config)

        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        self.classifier_dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)

    def init_classifier_bias(self, pos_counts, neg_counts):
        """ Trucul lui Karpathy: Inițializează bias-ul pentru a reflecta dezechilibrul claselor """
        bias_init = np.log((pos_counts + 1e-5) / (neg_counts + 1e-5))
        # Adăugăm argumentul `device` pentru a prelua locația actuală a stratului (GPU)
        self.classifier.bias.data = torch.tensor(
            bias_init,
            dtype=torch.float,
            device=self.classifier.weight.device
        )

    def forward(self, concept_ids, type_ids, time_vals, pos_ids, attention_mask):
        e_c = self.concept_emb(concept_ids)
        e_ty = self.type_emb(type_ids)
        e_p = self.pos_emb(pos_ids)
        e_ti = self.time_proj(self.time2vec(time_vals))

        x = e_c + e_ty + e_p + e_ti
        x = self.LayerNorm(x)
        x = self.embedding_dropout(x)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_outputs = self.encoder(x, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]

        cls_output = sequence_output[:, 0, :]
        pooled_output = self.pooler(cls_output)

        logits = self.classifier(self.classifier_dropout(pooled_output))
        return logits


# ============================================================
# 4. SCRIPT DE ANTRENARE CU MONITORIZARE
# ============================================================
def get_pos_weights(df, max_weight=15.0):
    """ Calculează pos_weight cu limite (clipping) pentru stabilitate """
    labels = df[LABEL_COLS].values
    num_positives = labels.sum(axis=0)
    num_negatives = len(df) - num_positives

    pos_weights = num_negatives / (num_positives + 1e-5)
    # Clipping pentru a nu lăsa o clasă extrem de rară să distrugă gradientul
    pos_weights = np.clip(pos_weights, a_min=1.0, a_max=max_weight)

    return torch.tensor(pos_weights, dtype=torch.float).to(device), num_positives, num_negatives


def train_model():
    print(f"--- Rulăm pe: {device} ---")
    df = pd.read_csv(CSV_PATH)
    vocab = build_vocab(df)
    vocab_size = len(vocab)
    print(f"Dimensiune vocabular medical: {vocab_size}")

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_loader = DataLoader(EHRDataset(train_df, vocab, MAX_SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EHRDataset(val_df, vocab, MAX_SEQ_LEN), batch_size=BATCH_SIZE)

    model = CEHRBERTModel(vocab_size=vocab_size, num_classes=13).to(device)

    # Calculăm weights și aplicăm Karpathy Trick
    pos_weights, pos_counts, neg_counts = get_pos_weights(train_df)
    model.init_classifier_bias(pos_counts, neg_counts)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Configurare Scheduler (Cosine cu Warmup)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)  # 10% din pași sunt de warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    print("\n--- ÎNCEPE ANTRENAREA CEHR-BERT ---")
    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0

        train_loop = tqdm(train_loader, desc=f"Epoca {epoch + 1}/{EPOCHS} [Train]", leave=False)
        for batch in train_loop:
            optimizer.zero_grad()

            c_ids = batch['concept_ids'].to(device)
            ty_ids = batch['type_ids'].to(device)
            tm_vals = batch['time_vals'].to(device)
            p_ids = batch['pos_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(c_ids, ty_ids, tm_vals, p_ids, mask)
            loss = criterion(logits, labels)

            loss.backward()

            # Gradient Clipping adăugat
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # Avansăm scheduler-ul

            total_loss += loss.item()
            # Afișăm și learning rate-ul curent în consolă
            current_lr = scheduler.get_last_lr()[0]
            train_loop.set_postfix(loss=loss.item(), lr=current_lr)

        # Faza de validare
        model.eval()
        val_loss = 0
        all_labels = []
        all_preds = []

        val_loop = tqdm(val_loader, desc=f"Epoca {epoch + 1}/{EPOCHS} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_loop:
                c_ids = batch['concept_ids'].to(device)
                ty_ids = batch['type_ids'].to(device)
                tm_vals = batch['time_vals'].to(device)
                p_ids = batch['pos_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(c_ids, ty_ids, tm_vals, p_ids, mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(logits)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        # Calcul Metrici
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)

        try:
            auc = roc_auc_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds > 0.5, average='weighted', zero_division=0)
        except ValueError:
            auc, f1 = 0, 0

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - start_time

        print(f"\nEpoca [{epoch + 1}/{EPOCHS}] | Timp: {epoch_time:.1f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ROC-AUC: {auc:.4f} | F1: {f1:.4f}")

        # Salvăm cel mai bun model bazat pe AUC
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), "cehr_bert_best_2.pth")
            print("  --> Model salvat! (Cel mai bun ROC-AUC)")

    print("\n--- ANTRENARE FINALIZATĂ ---")


if __name__ == "__main__":
    train_model()