# TP3 - Redes Neuronales Convolucionales

Clasificacion binaria de radiografias de torax en las clases `NORMAL` y `PNEUMONIA`, comparando dos arquitecturas preentrenadas con Transfer Learning:

- `VGG16`
- `ResNet50`

El objetivo es evaluar que modelo detecta mejor casos de neumonia y entender el impacto de decisiones como fine-tuning, data augmentation, class weights y metricas medicas.

## Dataset

Se utiliza el dataset **Chest X-Ray Images (Pneumonia)** de Kaggle.

Estructura esperada:

```text
dataset/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Distribucion usada:

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1349 | 3883 | 5232 |
| Test | 234 | 390 | 624 |

El dataset esta desbalanceado: en entrenamiento hay aproximadamente 2.9 imagenes de neumonia por cada imagen normal.

## Archivos principales

| Archivo | Descripcion |
|---------|-------------|
| `TP3_ChestXRay_VGG16_vs_ResNet50.ipynb` | Notebook principal: exploracion, entrenamiento, evaluacion y graficos |
| `informe_TP3.md` | Informe del trabajo practico |
| `presentacion_TP3.md` | Presentacion en formato Marp |
| `INSTRUCCIONES.md` | Guia de ejecucion y exportacion |
| `requirements.txt` | Dependencias Python |
| `resultados_comparacion.csv` | Metricas finales exportadas por el notebook |

## Instalacion

Crear un entorno virtual es recomendable:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Luego abrir el notebook:

```bash
jupyter notebook TP3_ChestXRay_VGG16_vs_ResNet50.ipynb
```

Ejecutar las celdas en orden desde el directorio raiz del proyecto. Keras descarga automaticamente los pesos de ImageNet de VGG16 y ResNet50 la primera vez.

## Metodologia

El entrenamiento se realiza en dos fases:

1. **Transfer Learning / Feature Extraction:** la base convolucional queda congelada y se entrena solo el clasificador agregado para `NORMAL` vs `PNEUMONIA`.
2. **Fine-Tuning:** se descongelan algunas capas finales de la base para adaptar caracteristicas de alto nivel al dominio de radiografias.

Tambien se usan:

- `ImageDataGenerator` con aumentacion moderada para reducir overfitting.
- `class_weight` para compensar el desbalance entre clases.
- `EarlyStopping` para detener el entrenamiento si `val_loss` deja de mejorar.
- `ModelCheckpoint` para guardar el mejor modelo.
- `ReduceLROnPlateau` para bajar el learning rate cuando la validacion se estanca.

## Resultados

Metricas finales sobre el conjunto de test:

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Especificidad |
|--------|----------|-----------|--------|----------|---------|---------------|
| VGG16 | 96.31% | 95.76% | 98.46% | 97.09% | 99.24% | 92.74% |
| ResNet50 | 93.75% | 98.08% | 91.79% | 94.83% | 98.46% | 97.01% |

Interpretacion breve:

- **VGG16** obtuvo mejor `Recall`, `F1-Score` y `AUC-ROC`, por lo que detecto mas casos reales de neumonia.
- **ResNet50** obtuvo mejor `Precision` y `Especificidad`, por lo que fue mas conservador al predecir neumonia y clasifico mejor los casos normales.
- En contexto medico, el `Recall` es especialmente importante porque mide cuantos pacientes con neumonia fueron detectados.

## Archivos generados

Al ejecutar el notebook completo se generan:

```text
distribucion_clases.png
muestras_dataset.png
curvas_entrenamiento.png
matrices_confusion.png
curvas_roc.png
comparacion_metricas.png
errores_vgg16.png
errores_resnet50.png
resultados_comparacion.csv
vgg16_best_tl.keras
vgg16_best_ft.keras
resnet50_best_tl.keras
resnet50_best_ft.keras
```

Los archivos `.keras` contienen los mejores modelos guardados durante entrenamiento. Si no aparecen, significa que todavia no se ejecutaron las celdas de entrenamiento correspondientes o no finalizaron correctamente.

## Como probar modelos ya entrenados

Para probar una imagen individual, debe existir el modelo entrenado (`vgg16_best_ft.keras` o `resnet50_best_ft.keras`). Ejemplo con VGG16:

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

print(f"Probabilidad de neumonia: {prob:.4f}")
print(f"Prediccion: {label}")
```

Para probar ResNet50:

```python
from tensorflow.keras.applications.resnet50 import preprocess_input

MODEL_PATH = "resnet50_best_ft.keras"
```

Usar siempre el `preprocess_input` correspondiente al modelo cargado. Si se carga VGG16, usar `vgg16.preprocess_input`; si se carga ResNet50, usar `resnet50.preprocess_input`.

## Nota importante

Este trabajo tiene fines academicos. Un modelo entrenado con este dataset no debe usarse como herramienta clinica real sin validacion externa, revision medica, analisis de sesgos e interpretabilidad.
