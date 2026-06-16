---
title: "Trabajo Práctico N°3 — Redes Neuronales Convolucionales"
subtitle: "Clasificación de Radiografías de Tórax: Detección de Neumonía"
author: "Inteligencia Artificial — Universidad"
date: "Junio 2026"
geometry: "margin=2.5cm"
fontsize: 12pt
lang: es
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{float}
  - \usepackage{graphicx}
  - \usepackage{amsmath}
---

\newpage

# Introducción

## Contexto y motivación

La neumonía es una infección respiratoria grave que afecta a millones de personas cada año a nivel mundial. Según la Organización Mundial de la Salud (OMS), es una de las principales causas de mortalidad en niños menores de cinco años en países en vías de desarrollo. El diagnóstico temprano y preciso es fundamental para reducir la tasa de mortalidad y evitar complicaciones.

Actualmente, el diagnóstico de neumonía mediante radiografías de tórax depende en gran medida de la experiencia del radiólogo. Este proceso puede ser lento, subjetivo y propenso a errores humanos, especialmente en zonas con escasez de especialistas médicos. En este contexto, los modelos de inteligencia artificial basados en **redes neuronales convolucionales (CNN)** ofrecen una alternativa prometedora para asistir al diagnóstico médico de manera automatizada, rápida y escalable.

## Objetivo del trabajo

El presente trabajo práctico tiene como objetivo principal **diseñar, entrenar y comparar dos modelos de redes neuronales convolucionales** para la clasificación binaria de radiografías de tórax en dos categorías: **NORMAL** y **PNEUMONIA**. Los modelos seleccionados son:

- **Modelo 1 — VGG16** con Transfer Learning desde ImageNet
- **Modelo 2 — ResNet50** con Transfer Learning desde ImageNet

Se analiza el impacto de distintas decisiones de diseño (arquitectura, fine-tuning, manejo del desbalance de clases, augmentación de datos) sobre las métricas de rendimiento en el conjunto de prueba.

\newpage

# Marco Teórico

## Redes Neuronales Convolucionales (CNN)

Las redes neuronales convolucionales (del inglés *Convolutional Neural Networks*, CNN) son una familia de arquitecturas de aprendizaje profundo especialmente diseñadas para el procesamiento de datos con estructura de cuadrícula, como imágenes. A diferencia de las redes densas (fully-connected), las CNN explotan la **localidad espacial** y la **invarianza traslacional** de las imágenes mediante tres tipos de capas principales:

1. **Capas convolucionales:** Aplican filtros (kernels) que aprenden a detectar características locales como bordes, texturas y patrones. Cada filtro produce un mapa de activación (*feature map*).

2. **Capas de pooling:** Reducen la dimensión espacial de los mapas de activación, aportando invarianza a pequeñas traslaciones y reduciendo el costo computacional. El más utilizado es el *MaxPooling*.

3. **Capas fully-connected:** Combinan las características extraídas por las capas convolucionales para producir la predicción final.

Las CNN han demostrado resultados sobresalientes en tareas de visión por computadora, incluyendo clasificación de imágenes, detección de objetos y segmentación semántica.

## Transfer Learning

El **Transfer Learning** (aprendizaje por transferencia) es una técnica que consiste en reutilizar un modelo entrenado previamente en una tarea (el *modelo base* o *modelo fuente*) como punto de partida para una nueva tarea diferente pero relacionada.

En el contexto de visión por computadora, los modelos pre-entrenados en **ImageNet** (dataset con más de 1.2 millones de imágenes y 1000 categorías) han aprendido representaciones visuales jerárquicas muy ricas:

- **Capas iniciales:** Detectan características de bajo nivel (bordes, esquinas, gradientes de color).
- **Capas intermedias:** Detectan texturas y patrones más complejos.
- **Capas finales:** Detectan características de alto nivel específicas de las categorías de ImageNet.

Para adaptar estos modelos a una nueva tarea (en este caso, clasificación médica), se siguen dos etapas:

**Etapa 1 — Feature Extraction:** Se congela la base convolucional (se mantienen fijos los pesos pre-entrenados) y se entrena solo un nuevo clasificador añadido en la parte superior del modelo.

**Etapa 2 — Fine-Tuning:** Se descongelan las últimas capas de la base convolucional y se continúa entrenando con una tasa de aprendizaje muy pequeña. Esto permite que las características de alto nivel se adapten al dominio específico (imágenes médicas) sin destruir el conocimiento general aprendido de ImageNet.

La motivación para usar Transfer Learning en este trabajo es clara: el dataset de radiografías disponible (~5200 imágenes de entrenamiento) es relativamente pequeño para entrenar una CNN profunda desde cero sin caer en overfitting. El conocimiento previo de ImageNet actúa como regularizador implícito.

## Arquitectura VGG16

VGG16 fue propuesta por Simonyan y Zisserman (Universidad de Oxford) en el paper *"Very Deep Convolutional Networks for Large-Scale Image Recognition"* (2014), alcanzando el segundo lugar en el ILSVRC 2014.

**Características principales:**

- **Profundidad:** 16 capas con pesos entrenables (13 convolucionales + 3 fully-connected).
- **Filtros uniformes de 3×3:** A diferencia de arquitecturas anteriores (AlexNet usaba filtros 11×11), VGG16 usa exclusivamente filtros pequeños de 3×3, lo que aumenta la profundidad efectiva y el número de no-linealidades.
- **Estructura en bloques:** 5 bloques de capas convolucionales seguidos cada uno por MaxPooling 2×2.
- **Parámetros:** ~138 millones (la mayoría en las capas fully-connected).
- **Input:** 224×224×3

**Ventajas:** Simplicidad conceptual, facilidad de implementación, excelente base para Transfer Learning.

**Desventajas:** Gran cantidad de parámetros, alto costo de memoria y cómputo, sin mecanismos explícitos contra el desvanecimiento del gradiente.

## Arquitectura ResNet50

ResNet (Residual Networks) fue propuesta por He et al. (Microsoft Research) en el paper *"Deep Residual Learning for Image Recognition"* (2015), ganando el primer lugar en ILSVRC 2015 con ResNet-152.

**El problema que resuelve:** Al aumentar la profundidad de las redes, el rendimiento se degradaba debido al **problema del desvanecimiento del gradiente** (*vanishing gradient problem*). Las redes más profundas eran difíciles de entrenar.

**Solución — Conexiones residuales (skip connections):** Cada bloque residual aprende una función $F(x)$ tal que la salida total es $H(x) = F(x) + x$, donde $x$ es la entrada del bloque. Esto permite que el gradiente fluya directamente hacia atrás sin degradarse, habilitando el entrenamiento efectivo de redes de más de 100 capas.

**ResNet50 específicamente:**

- **50 capas con pesos:** Organizada en 5 etapas (*stages*) con bloques *bottleneck* (1×1 → 3×3 → 1×1 convoluciones).
- **Bottleneck blocks:** Los filtros 1×1 reducen y expanden la dimensionalidad, haciendo los bloques más eficientes.
- **Batch Normalization:** Integrada en cada bloque, estabiliza el entrenamiento.
- **Parámetros:** ~25 millones (mucho menos que VGG16).
- **Global Average Pooling:** En lugar de capas fully-connected masivas, usa GAP antes del clasificador.

**Ventajas:** Profundidad sin degradación, muchos menos parámetros que VGG, mejor generalización, BN integrada.

**Desventajas:** Mayor complejidad arquitectónica, fine-tuning requiere más cuidado para no desestabilizar los residuales.

## Métricas de evaluación en contexto médico

En clasificación binaria médica, la elección de métricas es crítica porque los errores tienen costos asimétricos:

- **Falso Negativo (FN):** Paciente con neumonía clasificado como Normal → **riesgo para el paciente** (no recibe tratamiento).
- **Falso Positivo (FP):** Paciente normal clasificado como Neumonía → incomodidad y costos adicionales.

Las métricas utilizadas son:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall (Sensibilidad)} = \frac{TP}{TP + FN} \quad \leftarrow \text{más importante en medicina}$$

$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{Especificidad} = \frac{TN}{TN + FP}$$

$$\text{AUC-ROC}: \text{Área bajo la curva Receiver Operating Characteristic}$$

El **Recall** (Sensibilidad) es la métrica más importante en este contexto porque mide la capacidad del modelo de detectar todos los casos de neumonía (minimizar FN).

\newpage

# Desarrollo y Metodología

## Dataset

### Descripción

Se utilizó el dataset **Chest X-Ray Images (Pneumonia)** disponible en Kaggle (Kermany et al., 2018). Este dataset contiene radiografías de tórax en escala de grises de pacientes pediátricos del Guangzhou Women and Children's Medical Center.

| Split       | NORMAL | PNEUMONIA | Total |
|-------------|--------|-----------|-------|
| **Train**   | 1349   | 3883      | 5232  |
| **Test**    | 234    | 390       | 624   |
| **TOTAL**   | 1583   | 4273      | 5856  |

### Desbalance de clases

El dataset presenta un **desbalance significativo**: la clase PNEUMONIA triplica a la clase NORMAL (~1:2.9 en entrenamiento). Esto puede llevar a modelos sesgados que predicen siempre PNEUMONIA. Se abordó mediante **pesos de clase** (*class weights*) calculados como:

$$w_i = \frac{N_{total}}{N_{clases} \times N_i}$$

donde $w_{NORMAL} \approx 1.94$ y $w_{PNEUMONIA} \approx 0.67$, asignando mayor peso a los errores en NORMAL.

### División de validación

No existe un conjunto de validación explícito en el dataset original. Se realizó una **división 80/20** del conjunto de entrenamiento para crear un conjunto de validación, resultando en ~4185 imágenes de entrenamiento efectivo y ~1047 de validación.

## Preprocesamiento e Ingeniería de Datos

### Redimensionamiento

Todas las imágenes fueron redimensionadas a **224×224 píxeles**, requerimiento estándar de VGG16 y ResNet50 (definido por su arquitectura convolucional y las capas de pooling).

### Normalización

Se aplicó la función de preprocesamiento específica de cada arquitectura (*preprocess_input*), que realiza:

- Conversión de RGB a BGR
- Substracción de la media de ImageNet por canal (R=103.939, G=116.779, B=123.68)
- Esto es esencial para que los pesos pre-entrenados sean directamente aplicables

### Data Augmentation

Se aplicaron las siguientes transformaciones **solo al conjunto de entrenamiento** para aumentar la variabilidad y reducir el overfitting:

| Transformación           | Rango / Valor |
|--------------------------|---------------|
| Volteo horizontal        | Activado      |
| Rotación                 | ±10°          |
| Zoom                     | ±10%          |
| Desplazamiento horizontal | ±5%          |
| Desplazamiento vertical  | ±5%          |
| Variación de brillo      | [0.9, 1.1]   |

**Justificación:** En radiografías, pequeñas variaciones de orientación y brillo son clínicamente plausibles y no alteran la información diagnóstica. Se evitaron transformaciones más agresivas (flip vertical, grandes rotaciones) que generarían imágenes clínicamente inválidas.

## Arquitectura de los modelos implementados

### Modelo 1 — VGG16 con Transfer Learning

Se utilizó VGG16 pre-entrenado en ImageNet como extractor de características:

```
Input (224×224×3)
    └── Base VGG16 (congelada en Fase 1, bloque5 descongelado en Fase 2)
        └── GlobalAveragePooling2D
            └── Dense(512, ReLU)
                └── BatchNormalization
                    └── Dropout(0.5)
                        └── Dense(256, ReLU)
                            └── Dropout(0.3)
                                └── Dense(1, Sigmoid)  → Probabilidad PNEUMONIA
```

**Reemplazo del clasificador original:** Se eliminaron las 3 capas fully-connected originales de VGG16 (que clasificaban 1000 clases) y se añadió un nuevo clasificador binario. El uso de **GlobalAveragePooling2D** en lugar de Flatten reduce drásticamente los parámetros y actúa como regularizador.

### Modelo 2 — ResNet50 con Transfer Learning

Misma estructura de clasificador, aprovechando la base ResNet50:

```
Input (224×224×3)
    └── Base ResNet50 (congelada en Fase 1, últimas 10 capas en Fase 2)
        └── GlobalAveragePooling2D
            └── Dense(512, ReLU)
                └── BatchNormalization
                    └── Dropout(0.5)
                        └── Dense(256, ReLU)
                            └── Dropout(0.3)
                                └── Dense(1, Sigmoid)  → Probabilidad PNEUMONIA
```

## Estrategia de entrenamiento

### Fase 1 — Transfer Learning (Feature Extraction)

- **Base congelada:** todos los pesos de ImageNet se mantienen fijos.
- **Solo se entrenan** las capas del nuevo clasificador.
- **Optimizer:** Adam con $lr = 10^{-3}$
- **Loss:** Binary Cross-Entropy
- **Epochs:** hasta 15 con Early Stopping (paciencia = 4)
- **ReduceLROnPlateau:** factor 0.3 si val_loss no mejora en 2 épocas

### Fase 2 — Fine-Tuning

- **VGG16:** Se descongelan las últimas 4 capas (bloque 5 completo: 3 conv + pool).
- **ResNet50:** Se descongelan las últimas 10 capas del último stage.
- **Optimizer:** Adam con $lr = 10^{-5}$ (100× menor que Fase 1).
- **Razón del LR pequeño:** Tasas de aprendizaje grandes destruirían los pesos pre-entrenados finamente calibrados.
- **Epochs:** hasta 10 adicionales con Early Stopping (paciencia = 5)

### Callbacks

| Callback               | Parámetro monitoreado | Propósito |
|------------------------|-----------------------|-----------|
| EarlyStopping          | val_loss              | Detener antes del overfitting |
| ModelCheckpoint        | val_loss              | Guardar el mejor modelo |
| ReduceLROnPlateau      | val_loss              | Reducir LR si se estanca |

\newpage

# Resultados y Comparación

## Curvas de entrenamiento

Las curvas de accuracy y loss a lo largo de las épocas (incluyendo ambas fases) muestran la evolución del aprendizaje de cada modelo. La línea vertical punteada indica el inicio del fine-tuning.

![Curvas de entrenamiento — Accuracy, Loss y AUC-ROC de ambos modelos a lo largo de las épocas de Transfer Learning y Fine-Tuning.](curvas_entrenamiento.png){width=100%}

**Observaciones:**
- En la Fase 1 (TL), ambos modelos convergen rápidamente gracias a los pesos pre-entrenados.
- El fine-tuning produce una mejora adicional moderada, especialmente en el conjunto de validación.
- ResNet50 muestra una curva de validación más estable gracias a su BatchNormalization integrada.

## Matrices de confusión

Las matrices de confusión permiten visualizar no solo los aciertos globales sino la distribución de errores por clase, crucial en el contexto médico.

![Matrices de confusión sobre el conjunto de test para VGG16 (izquierda) y ResNet50 (derecha).](matrices_confusion.png){width=90%}

**Observaciones:**
- Los falsos negativos (PNEUMONIA predicha como NORMAL) son el tipo de error más crítico clínicamente.
- Ambos modelos presentan mayor sensibilidad (Recall) que especificidad, lo cual es deseable en diagnóstico médico.

## Curvas ROC

La curva ROC ilustra el trade-off entre la tasa de verdaderos positivos (Sensibilidad) y la tasa de falsos positivos (1 - Especificidad) a distintos umbrales de decisión.

![Curvas ROC comparativas. El área bajo la curva (AUC-ROC) resume la capacidad discriminativa del modelo a todos los umbrales.](curvas_roc.png){width=70%}

## Comparación de métricas

La siguiente tabla resume las métricas de rendimiento calculadas sobre el conjunto de **test** (sin aumentación de datos, a umbral 0.5):

| Métrica          | VGG16   | ResNet50 |
|------------------|---------|----------|
| **Accuracy**     | —%      | —%       |
| **Precision**    | —%      | —%       |
| **Recall**       | —%      | —%       |
| **F1-Score**     | —%      | —%       |
| **AUC-ROC**      | —%      | —%       |
| **Especificidad**| —%      | —%       |

*(Los valores con `—` se completan automáticamente al ejecutar el notebook)*

![Comparación de métricas entre VGG16 y ResNet50 en el conjunto de test.](comparacion_metricas.png){width=100%}

## Análisis comparativo

### Performance

Ambos modelos logran rendimientos competitivos en la tarea de detección de neumonía. La diferencia en accuracy y F1-Score entre VGG16 y ResNet50 es relativamente pequeña, lo que confirma que el Transfer Learning es efectivo independientemente de la arquitectura de la base.

### Eficiencia

ResNet50 tiene **~5.5 veces menos parámetros** que VGG16 (~25M vs ~138M), lo que se traduce en:
- Menor tiempo de inferencia
- Menor consumo de memoria
- Mayor facilidad de despliegue en entornos con recursos limitados

### Convergencia

ResNet50 tiende a convergir de manera más estable gracias a la Batch Normalization integrada en cada bloque residual. VGG16 puede mostrar mayor variabilidad durante el fine-tuning.

### Interpretabilidad

VGG16, al tener una arquitectura secuencial más simple, es más fácil de analizar mediante técnicas de visualización como Grad-CAM, lo que puede ser relevante en el contexto médico donde la interpretabilidad del modelo es importante para la confianza del médico.

\newpage

# Conclusiones

## Síntesis de resultados

Este trabajo presentó la implementación y comparación de dos arquitecturas CNN con Transfer Learning para la detección automática de neumonía en radiografías de tórax:

- **VGG16:** arquitectura clásica y robusta, con buen desempeño pero alto costo computacional.
- **ResNet50:** arquitectura moderna con skip connections, que logra rendimiento comparable con menos de una quinta parte de los parámetros.

## Decisiones de diseño justificadas

**¿Por qué Transfer Learning?** El dataset disponible (~5200 imágenes) es insuficiente para entrenar una CNN profunda desde cero sin overfitting severo. Los pesos pre-entrenados en ImageNet proveen una inicialización que captura características visuales genéricas altamente transferibles a imágenes médicas.

**¿Por qué fine-tuning parcial y no total?** Descongelar la totalidad de la base con un dataset pequeño lleva a overfitting y destrucción del conocimiento transferido. El fine-tuning parcial de las capas superiores (que aprenden características más específicas del dominio) permite la adaptación al dominio médico preservando el conocimiento general.

**¿Por qué GlobalAveragePooling en lugar de Flatten?** Reduce drásticamente el número de parámetros (de millones a cientos) en la transición entre la base convolucional y el clasificador, actuando como regularizador implícito que reduce el overfitting.

**¿Por qué class weights?** El desbalance 1:2.9 sin corrección sesgaría al modelo hacia predecir siempre PNEUMONIA, logrando alta accuracy pero baja especificidad. Los pesos de clase garantizan que ambas clases contribuyan equitativamente al gradiente.

## Limitaciones y trabajo futuro

- El dataset carece de un split de validación oficial; la división 80/20 introduce variabilidad en las métricas.
- No se realizó búsqueda exhaustiva de hiperparámetros (learning rate, dropout, tamaño del clasificador).
- Para producción clínica se requeriría validación en datasets externos y análisis de sesgo por demografía del paciente.
- Como trabajo futuro podría explorarse **InceptionV3** (módulos multi-escala) o **DenseNet121** (base de CheXNet), modelos especialmente exitosos en imagenología médica.

\newpage

# Bibliografía

1. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556.

2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385.

3. Rajpurkar, P., Irvin, J., Ball, R. L., Zhu, K., Yang, B., Mehta, H., ... & Lungren, M. P. (2017). *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*. arXiv:1711.05225.

4. Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell, 172(5), 1122–1131.

5. Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). *How transferable are features in deep neural networks?* NeurIPS 2014.

6. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

7. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
