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
├── README.md                                ← Guía general del proyecto
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

En CPU los tiempos pueden ser mucho mayores. En la ejecución observada, algunas épocas tardaron varios minutos.

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

## 4. Probar modelos ya entrenados

Para probar una imagen individual sin reentrenar, primero deben existir los archivos `.keras` generados por el notebook. El modelo VGG16 suele ser la opción recomendada si se prioriza detectar la mayor cantidad posible de neumonías; ResNet50 es más liviano y obtuvo mayor especificidad.

Ejemplo mínimo:

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

IMG_SIZE = (224, 224)
MODEL_PATH = "vgg16_best_ft.keras"
IMAGE_PATH = "dataset/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg"

model = keras.models.load_model(MODEL_PATH)

img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

prob = float(model.predict(x, verbose=0)[0][0])
label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"

print(f"Probabilidad de neumonía: {prob:.4f}")
print(f"Predicción: {label}")
```

Si se usa `resnet50_best_ft.keras`, cambiar el import por:

```python
from tensorflow.keras.applications.resnet50 import preprocess_input
MODEL_PATH = "resnet50_best_ft.keras"
```

Es importante usar el `preprocess_input` correspondiente al modelo cargado; VGG16 y ResNet50 esperan el mismo tamaño de imagen, pero deben recibir el preprocesamiento de su propia arquitectura.

## 5. Generar el PDF del informe

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

## 6. Notas importantes

- El informe y la presentación ya fueron actualizados con los valores de `resultados_comparacion.csv`.
- Las imágenes del informe se generan en el mismo directorio al ejecutar el notebook.
- El notebook usa `os.path.join` con rutas relativas — debe ejecutarse desde
  el directorio raíz del proyecto.
