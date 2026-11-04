#!/usr/bin/env python3
"""
FloodShift: Cross-Regional Generalization for Sentinel-1 Surface-Water Segmentation
Training and Evaluation Pipeline
"""

import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import uniform_filter
import tensorflow as tf
from tensorflow.keras import layers, models

# Global Leak-Free Training Statistics (Computed over valid pixels of familiar train regions)
TRAIN_STATS = {
    "vv_mean": -10.1397,
    "vv_std": 3.8141,
    "vv_p01": -22.50,
    "vv_p99": -1.50,
    "vh_mean": -16.9252,
    "vh_std": 4.3560,
    "vh_p01": -29.50,
    "vh_p99": -7.50,
}


def apply_lee_filter(band, window_size=5):
    """5x5 Local adaptive Lee speckle filter."""
    mean = uniform_filter(band, size=window_size)
    mean_sq = uniform_filter(band ** 2, size=window_size)
    local_var = np.maximum(mean_sq - mean ** 2, 0.0)
    global_var = np.var(band)
    w = local_var / (local_var + global_var + 1e-8)
    return mean + w * (band - mean)


def load_and_preprocess_chip(s1_path, label_path, mode="zscore", use_lee_filter=False, stats=TRAIN_STATS):
    """
    Loads dual-polarization Sentinel-1 chip and ground-truth mask.
    Handles invalid swath borders, applies requested normalization, and clamps invalid pixels to 0.0.
    """
    with rasterio.open(s1_path) as src_s1, rasterio.open(label_path) as src_lbl:
        sar = src_s1.read().astype(np.float32)  # (2, 512, 512)
        mask = src_lbl.read(1)                  # (512, 512)

    vv, vh = sar[0], sar[1]
    valid_mask = (mask != -1) & (~np.isnan(vv)) & (~np.isnan(vh)) & (~np.isinf(vv)) & (~np.isinf(vh))
    valid_mask_f = valid_mask.astype(np.float32)[..., np.newaxis]

    # Pre-filter imputation with mean to avoid window distortion
    vv_proc = np.where(valid_mask, vv, stats["vv_mean"])
    vh_proc = np.where(valid_mask, vh, stats["vh_mean"])

    if use_lee_filter:
        vv_proc = apply_lee_filter(vv_proc, window_size=5)
        vh_proc = apply_lee_filter(vh_proc, window_size=5)

    if mode == "raw_db":
        vv_norm = vv_proc
        vh_norm = vh_proc
    elif mode == "minmax_clip":
        vv_norm = (np.clip(vv_proc, -30.0, 0.0) + 30.0) / 30.0
        vh_norm = (np.clip(vh_proc, -30.0, 0.0) + 30.0) / 30.0
    elif mode == "zscore":
        vv_norm = (vv_proc - stats["vv_mean"]) / stats["vv_std"]
        vh_norm = (vh_proc - stats["vh_mean"]) / stats["vh_std"]
    elif mode == "percentile":
        vv_clip = np.clip(vv_proc, stats["vv_p01"], stats["vv_p99"])
        vh_clip = np.clip(vh_proc, stats["vh_p01"], stats["vh_p99"])
        vv_norm = (vv_clip - stats["vv_mean"]) / stats["vv_std"]
        vh_norm = (vh_clip - stats["vh_mean"]) / stats["vh_std"]
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    # Post-normalization zero-clamping
    vv_norm = np.where(valid_mask, vv_norm, 0.0)
    vh_norm = np.where(valid_mask, vh_norm, 0.0)

    X = np.stack([vv_norm, vh_norm], axis=-1)
    y = np.where(valid_mask, mask, 0.0).astype(np.float32)[..., np.newaxis]
    return X, y, valid_mask_f


def augment_chip(X, y, valid_mask, aug_mode="none"):
    """Applies geometric and/or radiometric intensity augmentation."""
    if aug_mode in ("geometric", "combined"):
        if np.random.rand() > 0.5:
            X, y, valid_mask = np.flip(X, axis=0), np.flip(y, axis=0), np.flip(valid_mask, axis=0)
        if np.random.rand() > 0.5:
            X, y, valid_mask = np.flip(X, axis=1), np.flip(y, axis=1), np.flip(valid_mask, axis=1)
        k = np.random.choice([0, 1, 2, 3])
        if k > 0:
            X, y, valid_mask = np.rot90(X, k, (0, 1)), np.rot90(y, k, (0, 1)), np.rot90(valid_mask, k, (0, 1))

    if aug_mode in ("intensity", "combined"):
        if np.random.rand() > 0.5:
            offset = np.random.uniform(-0.15, 0.15)
            noise = np.random.normal(0.0, 0.08, size=X.shape).astype(np.float32)
            X = np.where(valid_mask == 1.0, X + offset + noise, 0.0)

    return X, y, valid_mask


class InMemoryDataGenerator(tf.keras.utils.Sequence):
    """In-memory sequence generator for high-throughput training."""
    def __init__(self, file_list, s1_dir, label_dir, batch_size=2, norm_mode="zscore",
                 use_lee_filter=False, aug_mode="none", shuffle=True):
        self.batch_size = batch_size
        self.aug_mode = aug_mode
        self.shuffle = shuffle

        self.X_data = []
        self.y_data = []
        self.valid_data = []

        print(f"Pre-loading {len(file_list)} chips into RAM (norm={norm_mode}, lee={use_lee_filter})...")
        for f in file_list:
            s1_path = os.path.join(s1_dir, f)
            lbl_path = os.path.join(label_dir, f.replace("_S1Hand.tif", "_LabelHand.tif"))
            X, y, valid_mask = load_and_preprocess_chip(
                s1_path, lbl_path, mode=norm_mode, use_lee_filter=use_lee_filter
            )
            self.X_data.append(X)
            self.y_data.append(y)
            self.valid_data.append(valid_mask)

        self.indexes = np.arange(len(file_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X_data) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        batch_idxs = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X_batch, y_comb_batch = [], []

        for k in batch_idxs:
            X, y, valid_mask = self.X_data[k].copy(), self.y_data[k].copy(), self.valid_data[k].copy()
            if self.aug_mode != "none":
                X, y, valid_mask = augment_chip(X, y, valid_mask, aug_mode=self.aug_mode)

            y_combined = np.concatenate([y, valid_mask], axis=-1)
            X_batch.append(X)
            y_comb_batch.append(y_combined)

        return np.array(X_batch, dtype=np.float32), np.array(y_comb_batch, dtype=np.float32)


def masked_bce_dice_loss(y_true_comb, y_pred):
    """Masked Binary Cross-Entropy + Soft Dice Loss excluding invalid swath borders."""
    y_true = y_true_comb[..., 0:1]
    valid_mask = y_true_comb[..., 1:2]
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    bce = - (y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    masked_bce = tf.reduce_sum(bce * valid_mask) / (tf.reduce_sum(valid_mask) + 1e-7)

    intersection = tf.reduce_sum(valid_mask * y_true * y_pred)
    cardinality = tf.reduce_sum(valid_mask * (y_true + y_pred))
    masked_dice = 1.0 - (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
    return masked_bce + masked_dice


def masked_iou(y_true_comb, y_pred):
    """Batch-level masked IoU metric for ModelCheckpoint monitoring."""
    y_true = y_true_comb[..., 0:1]
    valid_mask = y_true_comb[..., 1:2]
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(valid_mask * y_true * y_pred_bin)
    union = tf.reduce_sum(valid_mask * y_true) + tf.reduce_sum(valid_mask * y_pred_bin) - intersection
    return (intersection + 1e-7) / (union + 1e-7)


def build_unet(input_shape=(512, 512, 2), base_filters=32):
    """Standard 2D U-Net encoder-decoder with BatchNormalization and ReLU activations."""
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    inputs = layers.Input(shape=input_shape)
    c1 = conv_block(inputs, base_filters); p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, base_filters * 2); p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, base_filters * 4); p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, base_filters * 8); p4 = layers.MaxPooling2D((2, 2))(c4)

    bn = conv_block(p4, base_filters * 16)

    u4 = layers.Concatenate()([layers.UpSampling2D((2, 2))(bn), c4]); d4 = conv_block(u4, base_filters * 8)
    u3 = layers.Concatenate()([layers.UpSampling2D((2, 2))(d4), c3]); d3 = conv_block(u3, base_filters * 4)
    u2 = layers.Concatenate()([layers.UpSampling2D((2, 2))(d3), c2]); d2 = conv_block(u2, base_filters * 2)
    u1 = layers.Concatenate()([layers.UpSampling2D((2, 2))(d2), c1]); d1 = conv_block(u1, base_filters)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(d1)
    return models.Model(inputs=inputs, outputs=outputs)


def evaluate_splits(model, s1_dir, label_dir, id_files, ood_files, norm_mode, use_lee_filter):
    """Computes exact dataset-level accumulated metrics across ID and OOD splits."""
    def accumulate_metrics(file_list):
        tp, fp, fn = 0.0, 0.0, 0.0
        for f in file_list:
            s1_path = os.path.join(s1_dir, f)
            lbl_path = os.path.join(label_dir, f.replace("_S1Hand.tif", "_LabelHand.tif"))
            X, y, valid_mask = load_and_preprocess_chip(s1_path, lbl_path, mode=norm_mode, use_lee_filter=use_lee_filter)
            pred = (model.predict(X[np.newaxis, ...], verbose=0)[0] > 0.5).astype(np.float32)
            tp += np.sum(y * pred * valid_mask)
            fp += np.sum((1.0 - y) * pred * valid_mask)
            fn += np.sum(y * (1.0 - pred) * valid_mask)
        iou = tp / (tp + fp + fn + 1e-7)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        return iou, dice, prec, rec

    id_iou, id_dice, id_prec, id_rec = accumulate_metrics(id_files)

    # Per-region OOD
    regional_metrics = {}
    for region in ["Bolivia", "Mekong", "Spain"]:
        r_files = [f for f in ood_files if f.startswith(region)]
        r_iou, _, _, _ = accumulate_metrics(r_files)
        regional_metrics[f"{region}_IoU"] = round(r_iou, 4)

    ood_iou, ood_dice, ood_prec, ood_rec = accumulate_metrics(ood_files)
    return {
        "ID_IoU": round(id_iou, 4),
        **regional_metrics,
        "OOD_IoU": round(ood_iou, 4),
        "OOD_Dice": round(ood_dice, 4),
        "OOD_Prec": round(ood_prec, 4),
        "OOD_Rec": round(ood_rec, 4),
        "Gap_ID_to_OOD": round(id_iou - ood_iou, 4)
    }


def main():
    parser = argparse.ArgumentParser(description="FloodShift Sentinel-1 U-Net Training and Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to HandLabeled directory containing S1Hand and LabelHand")
    parser.add_argument("--splits_dir", type=str, default="data_splits", help="Path to data splits folder")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Path to save weights and results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Resolve data path
    if args.data_dir is None:
        data_dir = r"C:\Users\hosam\OneDrive\Documents\PROJECT\sen1floods11\v1.1\data\flood_events\HandLabeled"
    else:
        data_dir = args.data_dir

    s1_dir = os.path.join(data_dir, "S1Hand")
    label_dir = os.path.join(data_dir, "LabelHand")

    # Load split filenames
    def read_split(fname):
        with open(os.path.join(args.splits_dir, fname), "r") as f:
            return [line.strip() for line in f if line.strip()]

    train_files = read_split("train_files_clean.txt")
    val_files = read_split("val_files.txt")
    id_files = read_split("id_test_files.txt")
    ood_files = read_split("ood_test_files.txt")

    exp_name = config.get("exp_name", "experiment")
    norm_mode = config.get("norm_mode", "zscore")
    use_lee = config.get("use_lee_filter", False)
    aug_mode = config.get("aug_mode", "none")
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 2)
    lr = config.get("lr", 1e-3)
    seed = config.get("seed", 42)

    print(f"\n{'='*70}\n[FloodShift] Running: {exp_name}")
    print(f"Norm: {norm_mode} | Aug: {aug_mode} | LeeFilter: {use_lee} | Epochs: {epochs} | Seed: {seed}\n{'='*70}")

    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(seed)

    train_gen = InMemoryDataGenerator(
        train_files, s1_dir, label_dir, batch_size=batch_size,
        norm_mode=norm_mode, use_lee_filter=use_lee, aug_mode=aug_mode, shuffle=True
    )
    val_gen = InMemoryDataGenerator(
        val_files, s1_dir, label_dir, batch_size=batch_size,
        norm_mode=norm_mode, use_lee_filter=use_lee, aug_mode="none", shuffle=False
    )

    model = build_unet(input_shape=(512, 512, 2), base_filters=32)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=masked_bce_dice_loss,
        metrics=[masked_iou]
    )

    weights_dir = os.path.join(args.output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, f"{exp_name}.keras")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=weight_path, monitor="val_masked_iou", mode="max", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_masked_iou", mode="max", patience=5, restore_best_weights=True, verbose=1
        )
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks, verbose=1)

    print("\nLoading best checkpoint for final evaluation...")
    model.load_weights(weight_path)
    results = evaluate_splits(model, s1_dir, label_dir, id_files, ood_files, norm_mode, use_lee)

    results_row = {"Experiment": exp_name, "Norm": norm_mode, "Aug": aug_mode, **results}
    print("\nEvaluation Results:")
    for k, v in results_row.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
