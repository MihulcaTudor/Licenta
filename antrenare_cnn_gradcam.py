import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
import torchxrayvision as xrv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

from torchmetrics import AUROC, Accuracy, Precision, Recall

# 1 HARDWARE SETUP
torch.set_float32_matmul_precision('medium')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("EROARE: Nu detectez placa video!")
    sys.exit()


# 2 DATASET & NORMALIZARE
class MIMICDataset(Dataset):
    def __init__(self, df, label_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.paths = self.df['image_path'].values
        self.labels = self.df[label_cols].values.astype(np.float32)
        self.study_ids = self.df['study_id'].astype(str).values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            image = np.array(Image.open(path).convert('L'))
        except Exception:
            image = np.zeros((224, 224), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)

        image = image.float()
        if image.ndim == 2:
            image = image.unsqueeze(0)

        image = image / 255.0
        image = image * 2048 - 1024

        # Returnez imaginea, etichetele si study_id-ul
        return image, torch.tensor(self.labels[idx]), self.study_ids[idx]


# 3 OVERSAMPLING LOGIC
def make_weighted_sampler(df, label_cols):
    print(" Calculare Oversampling Weights---------")
    temp_labels = df[label_cols].fillna(0).values
    class_counts = np.sum(temp_labels == 1, axis=0)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    num_samples = len(df)
    class_weights = num_samples / class_counts

    sample_weights = []
    for i in range(num_samples):
        labels = temp_labels[i]
        indices = np.where(labels == 1)[0]
        if len(indices) > 0:
            weight = np.max(class_weights[indices])
        else:
            weight = 0.2
        sample_weights.append(weight)

    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=num_samples, replacement=True
    )
    return sampler


# 4 MODEL & EXTRAGERE TRASATURI
class MIMICDenseNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        print("---Loading DenseNet121-MIMIC----")
        self.densenet = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
        self.densenet.op_threshs = None
        self.densenet.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.densenet(x)

    def extract_features(self, x):
        """Extrage penultimul strat (1024) si ultimul strat (13)"""
        with torch.no_grad():
            features = self.densenet.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))

            penultimate_vector = torch.flatten(out, 1)  # [Batch, 1024]
            logits_vector = self.densenet.classifier(penultimate_vector)  # [Batch, 13]

        return penultimate_vector, logits_vector


class MaskedBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        mask = ~torch.isnan(targets)
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        loss = self.bce(logits, safe_targets)
        loss = loss * mask.float()
        return loss.sum() / (mask.float().sum() + 1e-8)


# 4.1 GRAD-CAM IMPLEMENTATIONN
class DenseNetGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        # Ne legam de ultimul bloc dens al arhitecturii DenseNet121
        target_layer = self.model.densenet.features.norm5
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, target_class):
        self.model.eval()
        logits = self.model(input_image)
        self.model.zero_grad()

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_image.size(2), input_image.size(3)), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().detach().numpy()


# 5 antrenare
def train_epoch(model, loader, optimizer, criterion, scaler, epoch_idx, total_epochs):
    model.train()
    losses = []
    pbar = tqdm(loader, desc=f"Epoca {epoch_idx}/{total_epochs} [Train]", ncols=120)

    for images, labels, _ in pbar:  # Ignoram study_id aici
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        pbar.set_postfix({'loss': f"{np.mean(losses[-50:]):.4f}"})

    return np.mean(losses)


@torch.no_grad()
def validate(model, loader, criterion, metrics, epoch_idx, total_epochs):
    model.eval()
    losses = []
    for m in metrics.values(): m.reset()

    pbar = tqdm(loader, desc=f"Epoca {epoch_idx}/{total_epochs} [Valid]", ncols=120)

    for images, labels, _ in pbar:  # Ignoram study_id aici
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        losses.append(loss.item())
        probs = torch.sigmoid(outputs)
        mask = ~torch.isnan(labels)

        metrics['AUC'].update(probs, torch.nan_to_num(labels, nan=0).int())
        if mask.sum() > 0:
            metrics['Acc'].update(probs[mask], labels[mask].int())
            metrics['Prec'].update(probs[mask], labels[mask].int())
            metrics['Rec'].update(probs[mask], labels[mask].int())

        pbar.set_postfix({'val_loss': f"{np.mean(losses[-10:]):.4f}"})

    return (np.mean(losses), metrics['AUC'].compute().item(), metrics['Acc'].compute().item(),
            metrics['Prec'].compute().item(), metrics['Rec'].compute().item())


# 6. MAIN
if __name__ == '__main__':

    CSV_PATH = "mimic_complete_master_dataset.csv"
    FINAL_MODEL_FILE = "mimic_cnn_model_best_1.pth"

    BATCH_SIZE = 128
    IMAGE_SIZE = 224
    NUM_WORKERS = 4
    EPOCHS_PHASE_1 = 30
    EPOCHS_PHASE_2 = 20

    LABEL_COLS = [
        'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
        'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
    ]

    # 1 Prepare Data
    df_master = pd.read_csv(CSV_PATH)
    df_train = df_master[df_master['split'] == 'train']
    df_val = df_master[df_master['split'] == 'val']

    train_sampler = make_weighted_sampler(df_train, LABEL_COLS)

    train_aug = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        ToTensorV2(),
    ])
    val_aug = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), ToTensorV2()])

    train_loader = DataLoader(
        MIMICDataset(df_train, LABEL_COLS, train_aug),
        batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=3
    )

    val_loader = DataLoader(
        MIMICDataset(df_val, LABEL_COLS, val_aug),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # Loader pentru extragerea finala a tuturor datelor (fara augmentari, fara shuffle)
    all_data_loader = DataLoader(
        MIMICDataset(df_master, LABEL_COLS, val_aug),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # 2 Init Model si Tools
    model = MIMICDenseNet(len(LABEL_COLS)).to(DEVICE)
    criterion = MaskedBCE()
    scaler = torch.amp.GradScaler('cuda')

    metrics = {
        'AUC': AUROC(task="multilabel", num_labels=len(LABEL_COLS), average='macro').to(DEVICE),
        'Acc': Accuracy(task="binary").to(DEVICE),
        'Prec': Precision(task="binary").to(DEVICE),
        'Rec': Recall(task="binary").to(DEVICE)
    }

    best_auc_overall = 0.0

    # FAZA 1: Antrenare Classifier
    print(f"\n[FAZA 1] Antrenare Classifier ({EPOCHS_PHASE_1} Epoci)")
    for param in model.densenet.features.parameters(): param.requires_grad = False
    optimizer = optim.AdamW(model.densenet.classifier.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS_PHASE_1 + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, EPOCHS_PHASE_1)
        v_loss, v_auc, v_acc, v_prec, v_rec = validate(model, val_loader, criterion, metrics, epoch, EPOCHS_PHASE_1)
        print(f" -> Epoca {epoch}: Loss: {t_loss:.4f} | Val: {v_loss:.4f} | AUC: {v_auc:.4f}")

        if v_auc > best_auc_overall:
            best_auc_overall = v_auc
            torch.save(model.state_dict(), FINAL_MODEL_FILE)
            print(f" ->>> Model Salvat (AUC: {best_auc_overall:.4f})")

    # FAZA 2: Fine-Tuning
    print(f"\n[FAZA 2] Fine-Tuning ({EPOCHS_PHASE_2} Epoci)")
    if os.path.exists(FINAL_MODEL_FILE):
        model.load_state_dict(torch.load(FINAL_MODEL_FILE))

    for param in model.parameters(): param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader),
                                              epochs=EPOCHS_PHASE_2, pct_start=0.3)

    for epoch in range(1, EPOCHS_PHASE_2 + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, epoch, EPOCHS_PHASE_2)
        scheduler.step()
        v_loss, v_auc, v_acc, v_prec, v_rec = validate(model, val_loader, criterion, metrics, epoch, EPOCHS_PHASE_2)
        print(f" -> Epoca {epoch}: Loss: {t_loss:.4f} | Val: {v_loss:.4f} | AUC: {v_auc:.4f}")

        if v_auc > best_auc_overall:
            best_auc_overall = v_auc
            torch.save(model.state_dict(), FINAL_MODEL_FILE)
            print(f" ->>> Model Actualizat! (New Best AUC: {best_auc_overall:.4f})")

    print("\n[DONE] Antrenament complet.")

    # 7 EXTRAGERE SI SALVARE VECTORI (EMBEDDINGS)
    print("\n--- INCEPE EXTRAGEREA VECTORILOR (1024 si 13) PENTRU FUZIUNE OFFLINE---")
    model.load_state_dict(torch.load(FINAL_MODEL_FILE))
    model.eval()

    cnn_embeddings = {}
    extract_loop = tqdm(all_data_loader, desc="Extragere Vectori CNN")

    for images, _, study_ids in extract_loop:
        images = images.to(DEVICE)

        # Extragem vectorii
        vec_1024, vec_13 = model.extract_features(images)

        vec_1024_np = vec_1024.cpu().numpy()
        vec_13_np = vec_13.cpu().numpy()

        for i, s_id in enumerate(study_ids):
            # Daca un pacient are mai multe poze pe acelasi study_id, le vom salva ca o lista
            if s_id not in cnn_embeddings:
                cnn_embeddings[s_id] = {'image_features_1024': [], 'image_logits_13': []}

            cnn_embeddings[s_id]['image_features_1024'].append(vec_1024_np[i])
            cnn_embeddings[s_id]['image_logits_13'].append(vec_13_np[i])

    # Convertim listele in array-uri (daca are 2 poze, vectorul va fi o medie a lor)
    for s_id in cnn_embeddings:
        cnn_embeddings[s_id]['image_features_1024'] = np.mean(cnn_embeddings[s_id]['image_features_1024'], axis=0)
        cnn_embeddings[s_id]['image_logits_13'] = np.mean(cnn_embeddings[s_id]['image_logits_13'], axis=0)

    torch.save(cnn_embeddings, "cnn_embeddings_dict.pt")
    print("\nVectorii vizuali salvati in 'cnn_embeddings_dict.pt'")