import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torchxrayvision as xrv

from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef
)

from scipy.optimize import minimize
from scipy.special import logit, expit



# 1 CONFIGURARE
MODEL_PATH = "mimic_cnn_model_best_1.pth"
MASTER_CSV_PATH = Path("mimic_complete_master_dataset.csv")
IMAGE_SIZE = 224  # DenseNet XRV foloseste 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Lung Opacity', 'Lung Lesion',
    'Fracture', 'Support Devices', 'Enlarged Cardiomediastinum', 'Pleural Other'
]


def check_data_integrity(df, label_cols):
    print("\n" + "=" * 60)
    print("VERIFICARE DATASET (CATE NAN-URI AVEM?)")
    print("=" * 60)
    print(f"{'Patologie':<30} | {'Valid (0/1)':<15} | {'Ignorat (NaN)':<15}")
    print("-" * 80)
    for col in label_cols:
        nan_count = df[col].isna().sum()
        valid_count = len(df) - nan_count
        print(f"{col:<30} | {valid_count:<15} | {nan_count:<15}")
    print("-" * 80 + "\n")


# 2 DEFINITIE DATASET
class MIMICEvalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.paths = self.df['image_path'].values
        # Pastram NaN-urile
        self.labels = self.df[LABEL_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        try:
            # XRV vrea Grayscale ('L') -> Numpy Array
            image = np.array(Image.open(img_path).convert('L'))
        except:
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        # Preprocesare manuala (fara transforms complexe pt evaluare, doar resize si norm)
        # Resize manual daca nu folosim Albumentations aici
        if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
            img_pil = Image.fromarray(image).resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.array(img_pil)

        # Conversie la Tensor
        image = torch.tensor(image).float()

        # Adaugare canal dimensiune [1, 224, 224]
        if image.ndim == 2:
            image = image.unsqueeze(0)

        # Normalizare specifica XRV [-1024, 1024]
        image = image / 255.0
        image = image * 2048 - 1024

        return image, torch.tensor(self.labels[idx])


# 3 FUNCTIE PREDICTII
def get_predictions(model, loader):
    all_probs = []
    all_labels = []

    print(" Generare predictii")
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            # Folosim aceeasi precizie ca la antrenare
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                probs = torch.sigmoid(outputs)

            all_probs.append(probs.float().cpu().numpy())
            all_labels.append(labels.numpy())

    print(" Done.")
    # Concatenam toate batch-urile
    return np.vstack(all_labels), np.vstack(all_probs)


# 4. EXECUTIA PRINCIPALA
if __name__ == '__main__':

    if not MASTER_CSV_PATH.exists():
        print("EROARE: Nu gasesc master_dataset.csv")
        sys.exit()

    print(f"-----Incarcare date din {MASTER_CSV_PATH}--")
    df_master = pd.read_csv(MASTER_CSV_PATH)

    check_data_integrity(df_master, LABEL_COLS)

    if 'split' not in df_master.columns:
        print("EROARE: Coloana 'split' lipseste!")
        sys.exit()

    df_val = df_master[df_master['split'] == 'val'].copy()
    df_test = df_master[df_master['split'] == 'test'].copy()

    val_loader = DataLoader(MIMICEvalDataset(df_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(MIMICEvalDataset(df_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    print(f"----Incarcare model {MODEL_PATH}----")

    # 1 Definim arhitectura
    model = xrv.models.DenseNet(weights="densenet121-res224-mimic_ch")
    model.op_threshs = None  # Dezactivam pragurile vechi
    model.classifier = nn.Linear(1024, len(LABEL_COLS))  # Ajustam output-ul

    # 2 Incarcam ponderile antrenate
    try:

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "").replace("module.", "").replace("densenet.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print(" -> Model incarcat cu succes")
    except Exception as e:
        print(f"EROARE la incarcarea modelului: {e}")
        sys.exit()

    model.to(DEVICE)
    model.eval()

    print("\n--Calculare praguri optime pe Validare (IGNORAND NaN)---")
    y_true_val, y_pred_val_probs = get_predictions(model, val_loader)

    # PROBABILITY CALIBRATION (TEMPERATURE SCALING)
    print("\n------Aplicare Temperature Scaling pe Validare----")
    calibrators = {}
    y_pred_val_probs_calib = np.zeros_like(y_pred_val_probs)

    for i, class_name in enumerate(LABEL_COLS):
        col_true = y_true_val[:, i]
        col_pred = y_pred_val_probs[:, i]

        mask = ~np.isnan(col_true)
        valid_true = col_true[mask]
        valid_pred = col_pred[mask]

        if len(valid_true) > 0 and valid_true.sum() > 0:
            # 1 Recuperam logitii (z) din probabilitati (p) aplicand inversa functiei sigmoid
            # Limitam probabilitatile la extremitati ca sa evitam impartirea la zero
            valid_pred_clipped = np.clip(valid_pred, 1e-7, 1 - 1e-7)
            valid_logits = logit(valid_pred_clipped)


            # 2. Definim functia de cost (Binary Cross Entropy) pe care vrem sa o minimizam ajustand T
            def bce_loss(t_param):
                T = t_param[0]
                p_scaled = expit(valid_logits / T)
                # Formula Log-Loss
                loss = -np.mean(valid_true * np.log(p_scaled + 1e-7) + (1 - valid_true) * np.log(1 - p_scaled + 1e-7))
                return loss


            # 3. Optimizam Temperatura (T) plecand de la valoarea 1.0 (neutru)
            res = minimize(bce_loss, x0=[1.0], bounds=[(0.05, 10.0)])
            optimal_T = res.x[0]

            calibrators[class_name] = optimal_T
            y_pred_val_probs_calib[mask, i] = expit(valid_logits / optimal_T)
            print(f"  -> {class_name:<28} : T optim = {optimal_T:.4f}")
        else:
            calibrators[class_name] = 1.0
            y_pred_val_probs_calib[:, i] = col_pred

    # Suprascriem cu probabilitatile calibrate prin Temperatura
    y_pred_val_probs = y_pred_val_probs_calib

    # ----- Calcul Praguri (F1 Maxim) -----
    best_thresholds = {}


    threshold_data = []

    for i, class_name in enumerate(LABEL_COLS):
        col_true = y_true_val[:, i]
        col_pred = y_pred_val_probs[:, i]

        mask = ~np.isnan(col_true)
        valid_true = col_true[mask]
        valid_pred = col_pred[mask]

        if len(valid_true) == 0 or valid_true.sum() == 0:
            best_thresholds[class_name] = 0.5
            continue

        p, r, th = precision_recall_curve(valid_true, valid_pred)

        # Evitam impartirea la zero
        numerator = 2 * p * r
        denominator = p + r + 1e-7
        f1 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

        if len(th) == 0:
            best_thresholds[class_name] = 0.5
        else:
            best_idx = np.argmax(f1[:-1])  # Ultimul e 1.0
            best_thresholds[class_name] = float(th[best_idx])

        threshold_data.append([class_name, best_thresholds[class_name]])

    print(pd.DataFrame(threshold_data, columns=['Patologie', 'Best Threshold']).to_string(index=False))

    # ----- Predictii Finale pe Test ------
    print("\n----Evaluare pe Test (IGNORAND NaN)----")
    y_true_test, y_pred_test_probs = get_predictions(model, test_loader)

    # APLICARE TEMPERATURE SCALING PE SETUL DE TEST
    y_pred_test_probs_calib = np.zeros_like(y_pred_test_probs)
    for i, class_name in enumerate(LABEL_COLS):
        T = calibrators.get(class_name, 1.0)

        test_pred_clipped = np.clip(y_pred_test_probs[:, i], 1e-7, 1 - 1e-7)
        test_logits = logit(test_pred_clipped)

        y_pred_test_probs_calib[:, i] = expit(test_logits / T)

    y_pred_test_probs = y_pred_test_probs_calib




    for i, class_name in enumerate(LABEL_COLS):

        class_probs = y_pred_test_probs[:, i]

        c_min = np.min(class_probs)
        c_max = np.max(class_probs)
        c_mean = np.mean(class_probs)

        print(f"{class_name:<28} | {c_min:.4f}   | {c_max:.4f}   | {c_mean:.4f}")

    print("-" * 65 + "\n")

    # Binarizare
    y_pred_test_bin = np.zeros_like(y_pred_test_probs)
    for i, class_name in enumerate(LABEL_COLS):
        y_pred_test_bin[:, i] = (y_pred_test_probs[:, i] >= best_thresholds[class_name]).astype(int)
        # varianta 2 cu 0.5
        #y_pred_test_bin[:, i] = (y_pred_test_probs[:, i] >= 0.5).astype(int)

    # CALCUL METRICI CU FILTRARE
    results_data = []
    confusion_matrices = []

    micro_true = []
    micro_pred = []
    micro_prob = []

    for i, label in enumerate(LABEL_COLS):
        col_true = y_true_test[:, i]
        col_pred = y_pred_test_bin[:, i]
        col_prob = y_pred_test_probs[:, i]

        mask = ~np.isnan(col_true)
        valid_true = col_true[mask]
        valid_pred = col_pred[mask]
        valid_prob = col_prob[mask]

        micro_true.extend(valid_true)
        micro_pred.extend(valid_pred)
        micro_prob.extend(valid_prob)

        if len(valid_true) > 0:
            labels_present = [0, 1]
            # Calculam TN, FP, FN, TP manual daca lipseste vreo clasa
            cm = confusion_matrix(valid_true, valid_pred, labels=labels_present)
            tn, fp, fn, tp = cm.ravel()
            confusion_matrices.append(cm)

            p, r, f1, _ = precision_recall_fscore_support(valid_true, valid_pred, average='binary', zero_division=0)

            try:
                auc = roc_auc_score(valid_true, valid_prob)
            except:
                auc = 0.5

            try:
                pr_auc = average_precision_score(valid_true, valid_prob)
            except:
                pr_auc = 0.0

            kappa = cohen_kappa_score(valid_true, valid_pred)
            mcc = matthews_corrcoef(valid_true, valid_pred)

            if np.isnan(kappa): kappa = 0
            if np.isnan(mcc): mcc = 0

        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            p, r, f1, auc, pr_auc, kappa, mcc = 0, 0, 0, 0, 0, 0, 0
            confusion_matrices.append(np.zeros((2, 2)))

        results_data.append({
            'Patologie': label,
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'Precision': p, 'Recall': r, 'F1-Score': f1,
            'ROC-AUC': auc, 'PR-AUC': pr_auc,
            'Kappa': kappa, 'MCC': mcc
        })

    df_results = pd.DataFrame(results_data)

    # ----- MACRO AVERAGE -----
    macro_avg = df_results[['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'Kappa', 'MCC']].mean()
    macro_row = {
        'Patologie': 'MACRO AVERAGE',
        'TN': '-', 'FP': '-', 'FN': '-', 'TP': '-',
        'Precision': macro_avg['Precision'],
        'Recall': macro_avg['Recall'],
        'F1-Score': macro_avg['F1-Score'],
        'ROC-AUC': macro_avg['ROC-AUC'],
        'PR-AUC': macro_avg['PR-AUC'],
        'Kappa': macro_avg['Kappa'],
        'MCC': macro_avg['MCC']
    }

    # ----- MICRO AVERAGE -------
    micro_true = np.array(micro_true)
    micro_pred = np.array(micro_pred)
    micro_prob = np.array(micro_prob)

    p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(micro_true, micro_pred, average='binary', zero_division=0)
    try:
        auc_mic = roc_auc_score(micro_true, micro_prob)
    except:
        auc_mic = 0.5
    try:
        pr_mic = average_precision_score(micro_true, micro_prob)
    except:
        pr_mic = 0

    kappa_mic = cohen_kappa_score(micro_true, micro_pred)
    mcc_mic = matthews_corrcoef(micro_true, micro_pred)

    micro_row = {
        'Patologie': 'MICRO AVERAGE',
        'TN': '-', 'FP': '-', 'FN': '-', 'TP': '-',
        'Precision': p_mic,
        'Recall': r_mic,
        'F1-Score': f1_mic,
        'ROC-AUC': auc_mic,
        'PR-AUC': pr_mic,
        'Kappa': kappa_mic,
        'MCC': mcc_mic
    }

    df_final = pd.concat([df_results, pd.DataFrame([macro_row, micro_row])], ignore_index=True)

    print("\n" + "=" * 160)
    print(f"{'TABEL II: RAPORT COMPLET (IGNORAND NaN)':^160}")
    print("=" * 160)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(df_final.to_string(index=False, float_format="%.4f"))
    print("=" * 160)

    df_final.to_csv("rezultate_licenta_masked.csv", index=False)
    print("Raport CSV salvat: rezultate_licenta_masked.csv")

    # VIZUALIZARE HEATMAP GRID

    cols = 4
    rows = (len(LABEL_COLS) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.ravel()

    for i, label in enumerate(LABEL_COLS):
        cm = confusion_matrices[i]

        group_names = ['TN', 'FP', 'FN', 'TP']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

        # Calculam procentaje din totalul valid pt acea clasa
        total_valid = np.sum(cm)
        if total_valid > 0:
            group_percentages = ["{0:.2%}".format(value / total_valid) for value in cm.flatten()]
        else:
            group_percentages = ["0%", "0%", "0%", "0%"]

        labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=axes[i],
                    annot_kws={"size": 11, "weight": "bold"})

        axes[i].set_title(f"{label}", fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Predictie')
        axes[i].set_ylabel('Realitate')
        axes[i].set_xticklabels(['Neg', 'Poz'])
        axes[i].set_yticklabels(['Neg', 'Poz'])


    for j in range(len(LABEL_COLS), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle(f"Matrice de Confuzie (Excluzand NaN)", fontsize=20, fontweight='bold')
    plt.savefig("grid_matrici_confuzie_masked.png", dpi=300)
    print(" Imagine salvata: grid_matrici_confuzie_masked.png")
    plt.show()

    # 5 GRAD-CAM IMAGES FOR REPORT
    import cv2
    import types
    import torch.nn.functional as F


    class DenseNetGradCAM_Eval:
        def __init__(self, model):
            self.model = model
            self.gradients = None
            self.activations = None


            def patched_features2(self_model, x):
                features = self_model.features(x)
                out = F.relu(features, inplace=False)
                out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
                return out

            self.model.features2 = types.MethodType(patched_features2, self.model)

            target_layer = self.model.features.norm5
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
            cam = F.interpolate(cam, size=(input_image.size(2), input_image.size(3)), mode='bilinear',
                                align_corners=False)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            return cam.squeeze().cpu().detach().numpy()


    def generate_5_gradcams_for_report(model, test_loader, label_cols):


        grad_cam = DenseNetGradCAM_Eval(model)
        samples_generated = 0

        for idx in range(len(test_loader.dataset)):
            if samples_generated >= 5:
                break

            img_tensor, labels = test_loader.dataset[idx]

            positive_classes = torch.where(labels == 1.0)[0]
            if len(positive_classes) == 0:
                continue

            target_class_idx = positive_classes[0].item()
            target_class_name = label_cols[target_class_idx]

            input_image = img_tensor.unsqueeze(0).to(DEVICE)
            cam = grad_cam.generate_heatmap(input_image, target_class_idx)

            img_np = img_tensor.squeeze().cpu().numpy()
            img_np = (img_np + 1024) / 2048.0
            img_np = np.clip(img_np, 0, 1)

            cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255

            img_rgb = np.stack((img_np,) * 3, axis=-1)

            # Overlay
            cam_overlay = heatmap * 0.4 + img_rgb * 0.6
            cam_overlay = np.clip(cam_overlay, 0, 1)

            # Plotare DOAR overlay
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(cam_overlay)
            #ax.set_title(f"Grad-CAM Overlay (Pathology: {target_class_name})", fontsize=14, fontweight='bold')
            ax.axis('off')

            # Salvare
            save_name = f"gradcam_report_{samples_generated + 1}_{target_class_name.replace(' ', '_')}.png"
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            plt.close()

            print(f" Successfully saved: {save_name} (Focus on {target_class_name})")
            samples_generated += 1

        print("=" * 65 + "\n")


    generate_5_gradcams_for_report(model, test_loader, LABEL_COLS)

    # CONFUSION MATRIX HEATMAP GRID
    cols = 4
    rows = (len(LABEL_COLS) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.ravel()

    for i, label in enumerate(LABEL_COLS):
        cm = confusion_matrices[i]

        group_names = ['TN', 'FP', 'FN', 'TP']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

        total_valid = np.sum(cm)
        if total_valid > 0:
            group_percentages = ["{0:.2%}".format(value / total_valid) for value in cm.flatten()]
        else:
            group_percentages = ["0%", "0%", "0%", "0%"]

        labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)

        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=axes[i],
                    annot_kws={"size": 11, "weight": "bold"})

        axes[i].set_title(f"{label}", fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Prediction')
        axes[i].set_ylabel('True Label')
        axes[i].set_xticklabels(['Neg', 'Pos'])
        axes[i].set_yticklabels(['Neg', 'Pos'])


    for j in range(len(LABEL_COLS), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.suptitle("Confusion Matrix (Excluding NaN)", fontsize=20, fontweight='bold')
    plt.savefig("grid_confusion_matrices_masked.png", dpi=300)
    print(" Image saved: grid_confusion_matrices_masked.png")
    plt.show()