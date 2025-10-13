import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.colors as mcolors
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    cohen_kappa_score, recall_score
)
from dataset import FullImageDataset
from model.hsiclassifier import AdvancedHSIClassifier

MODEL_SAVE_FILENAME = "best_model.pth"
TRAINING_METRICS_FILENAME = "training_metrics.png"
CLASSIFICATION_REPORT_FILENAME = "classification_report.txt"
CLASS_MAP_BASE_FILENAME = "classification_map"

FULL_LOADER_BATCH_SIZE = 256
FULL_LOADER_NUM_WORKERS = 4
FULL_LOADER_PIN_MEMORY = True

PLOT_FIGSIZE = (15, 6)
GROUND_TRUTH_TITLE = "Ground Truth"
CLASS_RESULT_TITLE = "Classification Result"
TITLE_FONT_SIZE = 12
DPI = 300

BACKGROUND_COLOR = [0, 0, 0]
HUE_STEP = 30
SATURATION_BASE = 70
SATURATION_VARIANCE = 10
VALUE_BASE = 60
VALUE_VARIANCE = 15
CONTRAST_ENHANCE_FACTOR = 20

EVALUATION_TITLE = "Evaluation Results"
TITLE_SEPARATOR = "=" * 50

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    best_acc = 0.0
    train_losses, val_losses, accuracies = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({'batch_loss': loss.item()})

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        accuracies.append(accuracy)

        print(f'\nEpoch {epoch + 1}/{num_epochs} | '
              f'Train Loss: {epoch_train_loss:.4f} | '
              f'Val Loss: {epoch_val_loss:.4f} | '
              f'Val Accuracy: {accuracy:.4f}')

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_FILENAME)
            print(f'Best model updated (Acc: {best_acc:.4f})')

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    plt.subplot(122)
    plt.plot(range(1, num_epochs + 1), accuracies, label='Val Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.tight_layout()
    plt.savefig(TRAINING_METRICS_FILENAME, dpi=DPI, bbox_inches='tight')
    plt.close()

    return model


def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    oa = accuracy_score(all_labels, all_preds)
    class_recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)
    aa = np.mean(class_recalls)
    kappa = cohen_kappa_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(
        all_labels, all_preds, digits=4, zero_division=0
    )

    print(f'\n{TITLE_SEPARATOR}')
    print(EVALUATION_TITLE)
    print(TITLE_SEPARATOR)
    print(f'Overall Accuracy (OA): {oa:.4f}')
    print(f'Average Accuracy (AA): {aa:.4f}')
    print(f'Kappa Coefficient: {kappa:.4f}')
    print('\nConfusion Matrix:')
    print(conf_matrix)
    print('\nClassification Report:')
    print(class_report)

    with open(CLASSIFICATION_REPORT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(f'{TITLE_SEPARATOR}\n')
        f.write(f'{EVALUATION_TITLE}\n')
        f.write(f'{TITLE_SEPARATOR}\n')
        f.write(f'Overall Accuracy (OA): {oa:.4f}\n')
        f.write(f'Average Accuracy (AA): {aa:.4f}\n')
        f.write(f'Kappa Coefficient: {kappa:.4f}\n\n')
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(conf_matrix) + '\n\n')
        f.write('Classification Report:\n')
        f.write(class_report)

    return oa, aa, kappa, conf_matrix, class_report


def generate_classification_map(model, data, gt, patch_size, fa_model, device):
    full_dataset = FullImageDataset(data, gt, patch_size=patch_size, fa_model=fa_model)
    full_loader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=FULL_LOADER_BATCH_SIZE,
        shuffle=False,
        num_workers=FULL_LOADER_NUM_WORKERS,
        pin_memory=FULL_LOADER_PIN_MEMORY
    )

    height, width = gt.shape
    classification_map = np.zeros((height, width), dtype=np.uint8)

    model.to(device)
    model.eval()

    with torch.no_grad():
        for patches, i_coords, j_coords in tqdm(full_loader, desc='Generating Classification Map'):
            patches = patches.to(device)
            outputs = model(patches)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            i_coords = i_coords.numpy()
            j_coords = j_coords.numpy()

            for idx in range(len(preds)):
                i, j = i_coords[idx], j_coords[idx]
                classification_map[i, j] = preds[idx] + 1

    classification_map[gt == 0] = 0
    return classification_map


def visualize_classification_map(classification_map, gt):
    max_class = max(np.max(classification_map), np.max(gt))
    colors = [BACKGROUND_COLOR]
    for i in range(1, max_class + 1):
        hue = (i * HUE_STEP) % 360
        saturation = SATURATION_BASE + (i * SATURATION_VARIANCE) % 30
        value = VALUE_BASE + (i * VALUE_VARIANCE) % 40
        rgb = mcolors.hsv_to_rgb((hue / 360, saturation / 100, value / 100))
        colors.append([int(c * 255) for c in rgb])
    cmap = mcolors.ListedColormap(colors)

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.subplot(1, 2, 1)
    plt.imshow(gt, cmap=cmap, vmin=0, vmax=max_class)
    plt.title(GROUND_TRUTH_TITLE, fontsize=TITLE_FONT_SIZE)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(classification_map, cmap=cmap, vmin=0, vmax=max_class)
    plt.title(CLASS_RESULT_TITLE, fontsize=TITLE_FONT_SIZE)
    plt.axis('off')

    class_map_png = f'{CLASS_MAP_BASE_FILENAME}.png'
    plt.tight_layout()
    plt.savefig(class_map_png, dpi=DPI, bbox_inches='tight')
    plt.close()

    class_map_mat = f'{CLASS_MAP_BASE_FILENAME}.mat'
    scipy.io.savemat(class_map_mat, {'classification_map': classification_map})

    class_map_raw_png = f'{CLASS_MAP_BASE_FILENAME}_raw.png'
    img_data = classification_map.astype(np.float32)
    img_data[img_data > 0] = img_data[img_data > 0] * CONTRAST_ENHANCE_FACTOR
    img = Image.fromarray(img_data.astype(np.uint8))
    img.save(class_map_raw_png)

    print(f'Classification maps saved to:\n- {class_map_png}\n- {class_map_mat}\n- {class_map_raw_png}')
