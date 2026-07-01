"""
Script para regenerar todas las gráficas del TP3.
Carga los modelos FT ya entrenados y genera plots correctos.
NO reentrena nada.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              accuracy_score, precision_score,
                              recall_score, f1_score)

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

# ── Paths ──────────────────────────────────────────────────────────────────
TEST_DIR         = "dataset/chest_xray/test"
VGG16_FT_PATH    = "vgg16_best_ft.keras"
RESNET50_FT_PATH = "resnet50_best_ft.keras"
PNEUNET_PATH     = "pneunet_best.keras"
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 32

# ── Verifica que existen los modelos ──────────────────────────────────────
for p in [VGG16_FT_PATH, RESNET50_FT_PATH, PNEUNET_PATH]:
    assert os.path.exists(p), f"No se encontró: {p}"
    print(f"OK {p} ({os.path.getsize(p)/1e6:.1f} MB)")

# ── Generadores de test ────────────────────────────────────────────────────
def pneunet_preprocess(x):
    return x / 255.0

def make_test_gen(preprocess_fn):
    dg = ImageDataGenerator(preprocessing_function=preprocess_fn)
    return dg.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

test_vgg    = make_test_gen(vgg_preprocess)
test_resnet = make_test_gen(resnet_preprocess)
test_pneu   = make_test_gen(pneunet_preprocess)

print("\nClass indices:", test_vgg.class_indices)
print(f"Test samples: {test_vgg.n}")
n_normal    = (test_vgg.classes == 0).sum()
n_pneumonia = (test_vgg.classes == 1).sum()
print(f"  NORMAL={n_normal}, PNEUMONIA={n_pneumonia}")

# ── Carga modelos ──────────────────────────────────────────────────────────
print("\nCargando modelos…")
model_vgg    = keras.models.load_model(VGG16_FT_PATH,    compile=False)
model_resnet = keras.models.load_model(RESNET50_FT_PATH, compile=False)
model_pneu   = keras.models.load_model(PNEUNET_PATH,     compile=False)
print("OK Modelos cargados")

# ── Predicciones ───────────────────────────────────────────────────────────
def get_preds(model, test_gen):
    test_gen.reset()
    probs  = model.predict(test_gen, verbose=1).ravel()
    preds  = (probs > 0.5).astype(int)
    labels = test_gen.classes
    return labels, preds, probs

print("\nPrediciendo VGG16…")
y_true_vgg,    y_pred_vgg,    y_prob_vgg    = get_preds(model_vgg,    test_vgg)
print("Prediciendo ResNet50…")
y_true_resnet, y_pred_resnet, y_prob_resnet = get_preds(model_resnet, test_resnet)
print("Prediciendo PneuNet…")
y_true_pneu,   y_pred_pneu,   y_prob_pneu  = get_preds(model_pneu,   test_pneu)

# ── Función métricas ───────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, name):
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    auc_val = auc(fpr_arr, tpr_arr)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Accuracy    = {acc*100:.2f}%")
    print(f"  Precision   = {prec*100:.2f}%")
    print(f"  Recall      = {rec*100:.2f}%")
    print(f"  F1-Score    = {f1*100:.2f}%")
    print(f"  AUC         = {auc_val*100:.2f}%")
    print(f"  Specificity = {spec*100:.2f}%")
    return dict(Modelo=name, Accuracy=round(acc*100,2), Precision=round(prec*100,2),
                Recall=round(rec*100,2), F1=round(f1*100,2),
                AUC=round(auc_val*100,2), Especificidad=round(spec*100,2))

print("\n" + "="*40)
print("MÉTRICAS EN TEST SET")
m_vgg    = compute_metrics(y_true_vgg,    y_pred_vgg,    y_prob_vgg,    "VGG16")
m_resnet = compute_metrics(y_true_resnet, y_pred_resnet, y_prob_resnet, "ResNet50")
m_pneu   = compute_metrics(y_true_pneu,   y_pred_pneu,   y_prob_pneu,   "PneuNet")

# ── 1. Matrices de confusión ───────────────────────────────────────────────
print("\nGenerando matrices_confusion.png…")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Matrices de Confusión — Test Set", fontsize=14, fontweight='bold')

for ax, y_true, y_pred, title, cmap in [
    (axes[0], y_true_vgg,    y_pred_vgg,    "VGG16",    "Blues"),
    (axes[1], y_true_resnet, y_pred_resnet, "ResNet50", "Reds"),
    (axes[2], y_true_pneu,   y_pred_pneu,   "PneuNet",  "Greens"),
]:
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    labels = [["NORMAL", "PNEUMONIA"]]
    annot  = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, ax=ax,
                xticklabels=['NORMAL','PNEUMONIA'],
                yticklabels=['NORMAL','PNEUMONIA'],
                linewidths=0.5, cbar=False)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Real", fontsize=11)

plt.tight_layout()
plt.savefig('matrices_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK matrices_confusion.png")

# ── 2. Curvas ROC ──────────────────────────────────────────────────────────
print("Generando curvas_roc.png…")
fig, ax = plt.subplots(figsize=(7, 6))
for y_true, y_prob, name, color in [
    (y_true_vgg,    y_prob_vgg,    "VGG16",    "steelblue"),
    (y_true_resnet, y_prob_resnet, "ResNet50", "tomato"),
    (y_true_pneu,   y_prob_pneu,   "PneuNet",  "mediumseagreen"),
]:
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    auc_val = auc(fpr_arr, tpr_arr)
    ax.plot(fpr_arr, tpr_arr, label=f"{name} (AUC = {auc_val:.4f})", color=color, lw=2)

ax.plot([0,1],[0,1], 'k--', alpha=0.4, label="Clasificador aleatorio")
ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
ax.set_title("Curvas ROC — VGG16 vs ResNet50 vs PneuNet", fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('curvas_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK curvas_roc.png")

# ── 3. Comparación de métricas ─────────────────────────────────────────────
print("Generando comparacion_metricas.png…")
metricas = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Especificidad']
modelos  = [m_vgg, m_resnet, m_pneu]
colores  = ['steelblue', 'tomato', 'mediumseagreen']
nombres  = ['VGG16', 'ResNet50', 'PneuNet']

x     = np.arange(len(metricas))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 5))
for i, (m, col, nom) in enumerate(zip(modelos, colores, nombres)):
    vals  = [m[k] for k in metricas]
    bars  = ax.bar(x + i*width, vals, width, label=nom, color=col, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ymin = min(m[k] for m in modelos for k in metricas) - 5
ax.set_ylim(max(0, ymin), 105)
ax.set_xticks(x + width)
ax.set_xticklabels(metricas, fontsize=11)
ax.set_ylabel("Porcentaje (%)", fontsize=11)
ax.set_title("Comparación de Métricas — Test Set (3 modelos)", fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparacion_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK comparacion_metricas.png")

# ── 4. Falsos negativos (neumonías escapadas) ──────────────────────────────
print("Generando neumonias_escapadas2.png…")
fn_counts = []
fn_pcts   = []
total_pneu = n_pneumonia
for y_true, y_pred in [(y_true_vgg, y_pred_vgg),
                       (y_true_resnet, y_pred_resnet),
                       (y_true_pneu, y_pred_pneu)]:
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fn_counts.append(fn)
    fn_pcts.append(fn / total_pneu * 100)

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(nombres, fn_counts, color=colores, alpha=0.85, edgecolor='black', linewidth=0.5)
for bar, cnt, pct in zip(bars, fn_counts, fn_pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{cnt} neumonías\n({pct:.1f}%)", ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.set_ylabel("Cantidad de neumonías NO detectadas (FN)", fontsize=11)
ax.set_title(f"¿Cuántas neumonías se le escaparon a cada modelo?\n"
             f"(de {total_pneu} neumonías reales en test)", fontsize=12)
ax.set_ylim(0, max(fn_counts) * 1.3)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('neumonias_escapadas2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK neumonias_escapadas2.png")

# ── 5. Tamaño vs Accuracy ──────────────────────────────────────────────────
print("Generando tamanio_vs_accuracy2.png…")
params = {
    'VGG16':    model_vgg.count_params(),
    'ResNet50': model_resnet.count_params(),
    'PneuNet':  model_pneu.count_params(),
}
print("  Parámetros:", params)

fig, ax = plt.subplots(figsize=(8, 6))
for m, col, nom in zip(modelos, colores, nombres):
    acc  = m['Accuracy']
    rec  = m['Recall']
    p    = params[nom]
    size = rec / 5
    ax.scatter(p, acc, s=size**2, color=col, alpha=0.6, edgecolors='black', linewidth=0.8)
    ax.annotate(f"{nom}\n(rec={rec:.1f}%)", (p, acc),
                textcoords="offset points", xytext=(10, 5), fontsize=9)

ax.set_xscale('log')
ax.set_xlabel("Parámetros (escala log)", fontsize=11)
ax.set_ylabel("Accuracy en test (%)", fontsize=11)
ax.set_title("Trade-off: Tamaño vs Accuracy (burbuja = Recall)", fontsize=12)
ax.grid(True, alpha=0.3)

# Leyenda de burbujas
for col, nom in zip(colores, nombres):
    ax.scatter([], [], s=60, color=col, alpha=0.6, edgecolors='black', linewidth=0.8, label=nom)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('tamanio_vs_accuracy2.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK tamanio_vs_accuracy2.png")

print("\nTODAS LAS GRAFICAS GENERADAS CORRECTAMENTE")
print("="*50)
print("Resumen final:")
for m in [m_vgg, m_resnet, m_pneu]:
    print(f"  {m['Modelo']:10s}: Acc={m['Accuracy']:.1f}%, "
          f"Rec={m['Recall']:.1f}%, AUC={m['AUC']:.1f}%")
