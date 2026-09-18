"""
Unified Utilities and Multi-Class Evaluation Framework for Tree ML Lab
Optimized for 20-Class E-Commerce Tabular Benchmark with RTX 2050 GPU Acceleration.
"""

import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    log_loss, roc_auc_score, f1_score, accuracy_score,
    top_k_accuracy_score, confusion_matrix, classification_report
)

# Optional Intel CPU & GPU acceleration
try:
    from sklearnex import patch_sklearn
    patch_sklearn(verbose=False)
    INTEL_EXTENSION_AVAILABLE = True
except ImportError:
    INTEL_EXTENSION_AVAILABLE = False

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "None"
except ImportError:
    CUDA_AVAILABLE = False
    GPU_NAME = "None"


DEPARTMENTS_EN = [
    "Mobiles & Tablets", "Consumer Electronics", "Computers & Gaming",
    "Home & Kitchen", "Large Appliances", "Men's Fashion", "Women's Fashion",
    "Baby & Kids", "Beauty & Fragrances", "Personal Care & Health",
    "Supermarket & Groceries", "Sports & Fitness", "Automotive & Hardware",
    "Books & Stationery", "Toys & Games", "Furniture & Decor",
    "Watches & Jewelry", "Pet Supplies", "Office Electronics", "Travel & Luggage"
]

DEPARTMENTS_AR = [
    "هواتف وأجهزة لوحية", "إلكترونيات استهلاكية", "حواسيب وألعاب فيديو",
    "المنزل والمطبخ", "أجهزة منزلية كبرى", "أزياء رجالية", "أزياء نسائية",
    "أطفال ومواليد", "عطور ومستحضرات تجميل", "عناية شخصية وصحة",
    "سوبرماركت وبقالة", "رياضة ولياقة بدنية", "سيارات ومعدات صيانة",
    "كتب ومستلزمات مكتبية", "ألعاب ترفيهية", "أثاث وديكور منزلي",
    "ساعات ومجوهرات", "مستلزمات حيوانات أليفة", "تجهيزات مكتبية", "حقائب ومستلزمات سفر"
]


def print_hardware_summary():
    """Prints system resources and GPU capabilities."""
    print("=" * 80)
    print("HARDWARE & EXECUTION ENVIRONMENT")
    print("=" * 80)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"System RAM               : {ram_gb:.1f} GB")
    print(f"Intel Extension (oneDAL) : {'Active (Fast CPU)' if INTEL_EXTENSION_AVAILABLE else 'Not Installed'}")
    print(f"CUDA Available           : {CUDA_AVAILABLE}")
    print(f"GPU Model                : {GPU_NAME}")
    if CUDA_AVAILABLE:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM Capacity            : {vram_gb:.2f} GB (RTX 2050 Budget)")
    print("=" * 80)


def load_dataset(parquet_path=None, sample_rows=None, force_regenerate=False):
    """
    Loads dataset from Parquet searching standard repo paths,
    or generates a synthetic sample on the fly if file not found or force_regenerate=True.
    """
    resolved_path = None
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(pkg_dir, "data", "ecommerce_20class_sample_50k.parquet"),
        os.path.join(os.getcwd(), "data", "ecommerce_20class_sample_50k.parquet"),
        os.path.join(os.getcwd(), "..", "data", "ecommerce_20class_sample_50k.parquet"),
        os.path.join(os.getcwd(), "trees", "data", "ecommerce_20class_sample_50k.parquet"),
    ]
    
    if parquet_path and os.path.exists(parquet_path):
        resolved_path = parquet_path
    else:
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if resolved_path and os.path.exists(resolved_path) and not force_regenerate:
        df = pd.read_parquet(resolved_path)
        if sample_rows is not None and len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=42).reset_index(drop=True)
    else:
        # Generate inline leak-free sample if parquet not yet created on disk or forced
        try:
            from trees.data_generator import generate_chunk
        except ImportError:
            from data_generator import generate_chunk
        rows = sample_rows if sample_rows else 50_000
        df = generate_chunk(chunk_size=rows, num_features=200, seed=42)
        
        # Cache to disk for instant subsequent notebook runs
        save_target = resolved_path or candidates[0]
        try:
            os.makedirs(os.path.dirname(os.path.abspath(save_target)), exist_ok=True)
            df.to_parquet(save_target, compression='snappy')
            print(f"[Data Generator] Clean, leak-free dataset generated & cached to: {save_target}")
        except Exception:
            pass
        
    return df


def prepare_features(df, target_col='target', drop_cols=None):
    """
    Separates X and y, dropping target labels and string metadata.
    Handles categorical encoding for models that require numeric matrices.
    """
    if drop_cols is None:
        drop_cols = ['target', 'department_en', 'department_ar']
        
    y = df[target_col].values
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Store categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    return X, y, cat_cols, num_cols


def evaluate_multiclass_model(
    model_name,
    y_true,
    y_pred,
    y_prob=None,
    train_time=0.0,
    class_names=DEPARTMENTS_EN,
    y_train_true=None,
    y_train_pred=None,
    y_val_true=None,
    y_val_pred=None
):
    """
    Standard multi-class evaluation computing accuracy, Top-3, Macro F1, Log-Loss,
    Multi-Class ROC-AUC, and Train/Val/Test overfitting gap metrics.
    """
    test_acc = float(accuracy_score(y_true, y_pred))
    test_macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
    
    metrics = {
        'Model': model_name,
        'Training Time (s)': float(train_time),
        'Top-1 Accuracy': test_acc,
        'Top-3 Accuracy': np.nan,
        'Multi-Class Log-Loss': np.nan,
        'Multi-Class ROC-AUC (OvR)': np.nan,
        'Macro F1-Score': test_macro_f1,
        'Weighted F1-Score': float(f1_score(y_true, y_pred, average='weighted')),
    }
    
    # Train metrics & Overfitting diagnostics
    if y_train_true is not None and y_train_pred is not None:
        metrics['Train Accuracy'] = float(accuracy_score(y_train_true, y_train_pred))
        metrics['Train Macro-F1'] = float(f1_score(y_train_true, y_train_pred, average='macro'))
    else:
        metrics['Train Accuracy'] = np.nan
        metrics['Train Macro-F1'] = np.nan

    # Validation metrics
    if y_val_true is not None and y_val_pred is not None:
        metrics['Val Accuracy'] = float(accuracy_score(y_val_true, y_val_pred))
        metrics['Val Macro-F1'] = float(f1_score(y_val_true, y_val_pred, average='macro'))
    else:
        metrics['Val Accuracy'] = np.nan
        metrics['Val Macro-F1'] = np.nan

    # Explicit Overfitting Gap (Train Accuracy - Val Accuracy or Test Accuracy)
    if not np.isnan(metrics['Train Accuracy']):
        benchmark_eval = metrics['Val Accuracy'] if not np.isnan(metrics['Val Accuracy']) else metrics['Top-1 Accuracy']
        metrics['Overfitting Gap'] = float(metrics['Train Accuracy'] - benchmark_eval)
    else:
        metrics['Overfitting Gap'] = np.nan

    # Top-3 Accuracy and probability metrics
    if y_prob is not None:
        metrics['Top-3 Accuracy'] = float(top_k_accuracy_score(y_true, y_prob, k=3, labels=np.arange(len(class_names))))
        try:
            metrics['Multi-Class Log-Loss'] = float(log_loss(y_true, y_prob))
            metrics['Multi-Class ROC-AUC (OvR)'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        except Exception:
            pass
            
    return metrics


def evaluate_model_splits(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    model_name="Model",
    train_time=0.0,
    class_names=DEPARTMENTS_EN,
    precomputed=None
):
    """
    Convenience function: generates predictions across Train, Validation, and Test sets,
    and returns comprehensive metrics including the Overfitting Gap.

    Parameters
    ----------
    precomputed : dict, optional
        Pre-computed predictions to use instead of calling model.predict().
        Useful for GPU models (e.g. XGBoost) to avoid device-mismatch warnings.
        Expected format:
            {
                'train': (y_train_pred_array, y_train_prob_array_or_None),
                'val':   (y_val_pred_array,   y_val_prob_array_or_None),
                'test':  (y_test_pred_array,  y_test_prob_array_or_None),
            }
    """
    if precomputed is not None:
        y_train_pred, _           = precomputed.get('train', (None, None))
        y_val_pred,   _           = precomputed.get('val',   (None, None))
        y_test_pred,  y_test_prob = precomputed.get('test',  (None, None))
    else:
        y_train_pred = model.predict(X_train)
        y_val_pred   = model.predict(X_val)
        y_test_pred  = model.predict(X_test)

        y_test_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_test_prob = model.predict_proba(X_test)
            except Exception:
                y_test_prob = None

    return evaluate_multiclass_model(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_test_pred,
        y_prob=y_test_prob,
        train_time=train_time,
        class_names=class_names,
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_val_true=y_val,
        y_val_pred=y_val_pred
    )


def plot_overfitting_analysis(depths, train_scores, val_scores, metric_name="Accuracy", title="Overfitting Analysis: Train vs. Validation"):
    """
    Plots a dual learning curve showing where the model begins to overfit.
    """
    plt.figure(figsize=(10, 5))
    x_labels = [str(d) if d is not None else 'None' for d in depths]
    plt.plot(x_labels, train_scores, marker='o', linewidth=2, color='#1f77b4', label=f'Train {metric_name}')
    plt.plot(x_labels, val_scores, marker='s', linewidth=2, color='#ff7f0e', label=f'Val {metric_name}')
    plt.title(title, fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Tree Max Depth', fontsize=11)
    plt.ylabel(metric_name, fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_20(y_true, y_pred, class_names=DEPARTMENTS_EN, title="20-Class Confusion Matrix"):
    """
    Plots a 20x20 normalized confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm, annot=False, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Normalized Prediction Ratio'}
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Department', fontsize=11, fontweight='bold')
    plt.ylabel('True Department', fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(results_list, metric_name='Top-1 Accuracy'):
    """
    Plots horizontal bar chart comparing models on a specified metric.
    """
    df = pd.DataFrame(results_list)
    if metric_name not in df.columns:
        return
        
    df_sorted = df.sort_values(metric_name, ascending=True)
    plt.figure(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(df_sorted))
    bars = plt.barh(df_sorted['Model'], df_sorted[metric_name], color=colors, edgecolor='black', alpha=0.85)
    plt.title(f'Model Comparison: {metric_name}', fontsize=13, fontweight='bold')
    plt.xlabel(metric_name, fontsize=11, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.005, bar.get_y() + bar.get_height()/2, f'{w:.4f}', va='center', fontsize=10)
        
    plt.tight_layout()
    plt.show()
