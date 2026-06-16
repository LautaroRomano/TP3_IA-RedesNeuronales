# TP3 — Instrucciones de Ejecución

## Estructura del proyecto

```
TP3_IA RedesNeuronales/
├── dataset/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/      (1349 imágenes)
│       │   └── PNEUMONIA/   (3883 imágenes)
│       └── test/
│           ├── NORMAL/      (234 imágenes)
│           └── PNEUMONIA/   (390 imágenes)
├── TP3_ChestXRay_VGG16_vs_ResNet50.ipynb   ← Notebook principal
├── informe_TP3.md                           ← Informe (convertir a PDF)
├── presentacion_TP3.md                      ← Presentación (Marp)
├── requirements.txt
└── INSTRUCCIONES.md
```

## 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 2. Ejecutar el notebook

```bash
jupyter notebook TP3_ChestXRay_VGG16_vs_ResNet50.ipynb
```

Ejecutar las celdas **en orden**. El notebook descarga automáticamente los pesos
de VGG16 y ResNet50 desde los servidores de Keras (~600MB total, solo la primera vez).

### Tiempo estimado de entrenamiento (GPU NVIDIA):
- Fase TL VGG16: ~8-12 minutos
- Fine-Tuning VGG16: ~5-8 minutos
- Fase TL ResNet50: ~6-10 minutos
- Fine-Tuning ResNet50: ~4-7 minutos

## 3. Archivos generados tras ejecutar el notebook

| Archivo | Descripción |
|---------|-------------|
| `curvas_entrenamiento.png`  | Accuracy/Loss/AUC por época |
| `matrices_confusion.png`    | Matrices de confusión |
| `curvas_roc.png`            | Curvas ROC comparativas |
| `comparacion_metricas.png`  | Barras de métricas |
| `distribucion_clases.png`   | Distribución del dataset |
| `muestras_dataset.png`      | Ejemplos de imágenes |
| `resultados_comparacion.csv`| Tabla de métricas en CSV |
| `vgg16_best_ft.keras`       | Pesos VGG16 (mejor modelo) |
| `resnet50_best_ft.keras`    | Pesos ResNet50 (mejor modelo) |

## 4. Generar el PDF del informe

### Opción A — Pandoc (recomendado):
```bash
# Instalar pandoc desde https://pandoc.org/installing.html
pandoc informe_TP3.md -o informe_TP3.pdf --pdf-engine=pdflatex
```

Si hay error con fuentes, usar:
```bash
pandoc informe_TP3.md -o informe_TP3.pdf --pdf-engine=xelatex
```

### Opción B — Marp CLI (para la presentación):
```bash
npm install -g @marp-team/marp-cli
marp presentacion_TP3.md --pdf -o presentacion_TP3.pdf
marp presentacion_TP3.md --pptx -o presentacion_TP3.pptx
```

### Opción C — VS Code:
- Instalar extensión "Marp for VS Code"
- Abrir `presentacion_TP3.md` → Export as PDF

### Opción D — Copiar a Word:
Copiar el contenido del `.md` a Word y formatear manualmente.

## 5. Notas importantes

- El informe tiene marcadores `—%` en la tabla de resultados.
  **Reemplazar con los valores reales del CSV** tras ejecutar el notebook.
- Las imágenes del informe se generan en el mismo directorio al ejecutar el notebook.
- El notebook usa `os.path.join` con rutas relativas — debe ejecutarse desde
  el directorio raíz del proyecto.
