import torch
import torch.optim as optim
import numpy as np
import scipy.io
import os
from torch.utils.data import Subset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FactorAnalysis

from dataset import HSI_Dataset, find_hsi_key, find_gt_key
from model.hsiclassifier import AdvancedHSIClassifier
from model.core_modules import FocalLoss
from train import (
    train_model, evaluate_model,
    generate_classification_map, visualize_classification_map
)


def main():
    DATA_PATH = "${DATA_PATH}"
    PATCH_SIZE = ${PATCH_SIZE}
    BATCH_SIZE = ${BATCH_SIZE}
    NUM_EPOCHS = ${NUM_EPOCHS}
    LR = ${LEARNING_RATE}
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TRAIN_TEST_SPLIT = ${TRAIN_TEST_SPLIT}
    FA_COMPONENTS_RATIO = ${FA_RATIO}
    MODEL_SAVE_PATH = "${MODEL_SAVE_PATH}"
    CLASSIFICATION_MAP_PATH = "${CLASSIFICATION_MAP_PATH}"

    print(f'Using Device: {DEVICE}')
    print(f'Dataset Path: {DATA_PATH}')
    print('Loading HSI Dataset...')

    try:
        data_mat = scipy.io.loadmat(os.path.join(DATA_PATH, "${HSI_DATA_FILE}"))
        gt_mat = scipy.io.loadmat(os.path.join(DATA_PATH, "${HSI_GT_FILE}"))
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found - {e}")
        print(f"Please check if '${HSI_DATA_FILE}' and '${HSI_GT_FILE}' exist in {DATA_PATH}")
        return
    except Exception as e:
        print(f"Error loading dataset - {e}")
        return

    data_key = ${DATA_KEY} if "${DATA_KEY}" else find_hsi_key(data_mat)
    gt_key = ${GT_KEY} if "${GT_KEY}" else find_gt_key(gt_mat)
    if gt_key is None:
        gt_key = find_gt_key(data_mat)

    print(f'Detected Data Key: {data_key}')
    print(f'Detected GT Key: {gt_key}')

    if data_key is None or gt_key is None:
        print("Error: Failed to find data/gt key!")
        print("Please manually set 'DATA_KEY' and 'GT_KEY' identifiers")
        return

    try:
        data = data_mat[data_key]
        labels = gt_mat[gt_key] if gt_key in gt_mat else data_mat[gt_key]
    except KeyError as e:
        print(f"Error: Key {e} not found! Please check 'DATA_KEY' and 'GT_KEY' identifiers")
        return

    if data.ndim == 3:
        if data.shape[0] < data.shape[-1]:
            data = np.transpose(data, (1, 2, 0))
    else:
        print(f"Error: Data dimension error (expected 3D, got {data.ndim}D)")
        return

    if data.shape[0] != labels.shape[0] or data.shape[1] != labels.shape[1]:
        raise ValueError(
            f"Spatial dimension mismatch!\n"
            f"Data: {data.shape[:2]}, GT: {labels.shape}"
        )

    height, width, num_bands = data.shape
    print(f'Dataset Info: {height}×{width} Spatial Size, {num_bands} Bands')

    print('Applying Standardization (Z-score)...')
    data_flat = data.reshape(-1, num_bands)
    scaler = StandardScaler()
    data_flat = scaler.fit_transform(data_flat)
    data = data_flat.reshape(height, width, num_bands)

    valid_mask = labels != 0
    valid_pixels = data[valid_mask]
    valid_labels = labels[valid_mask]
    NUM_CLASSES = len(np.unique(valid_labels))
    print(f'Valid Classes: {NUM_CLASSES} (Classes: {np.unique(valid_labels)})')

    if NUM_CLASSES < 2:
        print("Error: Less than 2 valid classes! Cannot perform classification.")
        return

    FA_COMPONENTS = max(${MIN_FA_COMPONENTS}, num_bands // FA_COMPONENTS_RATIO)
    print(f'Applying Factor Analysis (Components: {FA_COMPONENTS})...')
    train_idx_fa, _ = train_test_split(
        np.arange(len(valid_pixels)), test_size=${FA_TEST_SIZE}, random_state=${RANDOM_SEED}, stratify=valid_labels
    )
    fa = FactorAnalysis(n_components=FA_COMPONENTS, random_state=${RANDOM_SEED})
    fa.fit(valid_pixels[train_idx_fa])

    data_flat = data.reshape(-1, num_bands)
    data_fa = fa.transform(data_flat)
    data_fa = data_fa.reshape(height, width, FA_COMPONENTS)
    print(f'Reduced Data Shape: {data_fa.shape} (H×W×Reduced Bands)')

    full_dataset = HSI_Dataset(
        data=data_fa, labels=labels, patch_size=PATCH_SIZE, is_train=False, fa_model=None
    )

    class_counts = np.zeros(NUM_CLASSES)
    for _, label in full_dataset:
        if 0 <= label < NUM_CLASSES:
            class_counts[label] += 1
    class_weights = 1.0 / (np.sqrt(class_counts) + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    sample_weights = np.array([class_weights[label] for _, label in full_dataset])

    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=1 - TRAIN_TEST_SPLIT,
        random_state=${RANDOM_SEED},
        stratify=[full_dataset.labels[i, j] for i, j in full_dataset.valid_indices]
    )

    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_sample_weights = sample_weights[train_idx]
    sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_sample_weights),
        replacement=${SAMPLER_REPLACEMENT}
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=${PIN_MEMORY}
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=${PIN_MEMORY}
    )

    print(f'Dataset Split: Train={len(train_dataset)} samples, Test={len(test_dataset)} samples')
    print(f'Train Loader: {len(train_loader)} batches, Test Loader: {len(test_loader)} batches')

    print('Initializing Model...')
    model = AdvancedHSIClassifier(
        in_channels=${IN_CHANNELS},
        spatial_size=PATCH_SIZE,
        num_bands=FA_COMPONENTS,
        num_classes=NUM_CLASSES
    )

    alpha = torch.tensor([${FOCAL_LOSS_ALPHA}] * NUM_CLASSES, dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=${FOCAL_LOSS_GAMMA})

    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=${WEIGHT_DECAY}
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=${SCHEDULER_MODE}, factor=${SCHEDULER_FACTOR}, patience=${SCHEDULER_PATIENCE}
    )

    print('Starting Model Training...')
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        device=DEVICE
    )

    print('\nStarting Model Evaluation...')
    oa, aa, kappa, conf_matrix, class_report = evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        device=DEVICE
    )

    print('\n' + '=' * 60)
    print('Final Evaluation Metrics')
    print('=' * 60)
    print(f'Overall Accuracy (OA): {oa:.4f}')
    print(f'Average Accuracy (AA): {aa:.4f}')
    print(f'Kappa Coefficient: {kappa:.4f}')
    print(f'OA - AA Difference: {abs(oa - aa):.4f}')
    print('=' * 60)

    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f'Final model saved as "{MODEL_SAVE_PATH}"')

    print('\nGenerating Full-Image Classification Map...')
    classification_map = generate_classification_map(
        model=trained_model,
        data=data,
        gt=labels,
        patch_size=PATCH_SIZE,
        fa_model=fa,
        device=DEVICE
    )

    visualize_classification_map(
        classification_map=classification_map,
        gt=labels,
        save_path=CLASSIFICATION_MAP_PATH
    )

    print('\nAll Tasks Completed!')


if __name__ == '__main__':
    main()
