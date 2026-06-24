---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
color: #2c3e50
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 28px;
  }
  h1 { color: #1a5276; border-bottom: 3px solid #2980b9; padding-bottom: 8px; }
  h2 { color: #2471a3; }
  strong { color: #c0392b; }
  table { font-size: 22px; }
  img { border-radius: 8px; }
  .highlight { background: #eaf2ff; padding: 10px; border-left: 4px solid #2980b9; border-radius: 4px; }
---

# TP3 — Redes Neuronales Convolucionales
## Detección de Neumonía en Radiografías de Tórax
### VGG16 vs ResNet50 vs PneuNet

**Inteligencia Artificial | Junio 2026**

---

# Agenda

1. Problema y motivación
2. Dataset: Chest X-Ray
3. Conceptos: CNN y Transfer Learning
4. Arquitecturas: VGG16, ResNet50 y **PneuNet**
5. Metodología implementada
6. Resultados y comparación
7. Conclusiones

---

# El Problema

## ¿Por qué detectar neumonía con IA?

- La neumonía es una de las **principales causas de mortalidad infantil** (OMS)
- Diagnóstico actual: radiografías interpretadas por radiólogos
- **Problemas:** escasez de especialistas, subjetividad, tiempo

> **Objetivo:** Clasificar radiografías de tórax como **NORMAL** o **PNEUMONIA** de forma automática usando redes neuronales convolucionales.

---

# Dataset: Chest X-Ray

## Kermany et al. (2018) — Kaggle

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1.349  | 3.883     | 5.232 |
| Test  | 234    | 390       | 624   |

## Desafíos:
- **Desbalance 1:2.9** → clase NORMAL sub-representada
- Dataset pequeño para entrenar desde cero
- Imágenes médicas: requieren alta sensibilidad

---

# Ejemplos del Dataset

![w:900](muestras_dataset.png)

*Superior: radiografías NORMALES — Inferior: radiografías con NEUMONÍA*

---

# Distribución de Clases

![w:800](distribucion_clases.png)

**Solución al desbalance:** Class Weights en la función de pérdida
$$w_{NORMAL} = 1.94 \qquad w_{PNEUMONIA} = 0.67$$

---

# ¿Qué es Transfer Learning?

## Aprovechar conocimiento previo

```
ImageNet (1.2M imágenes, 1000 clases)
         ↓  Pre-entrenamiento
     Modelo Base  →  Características generales aprendidas
         ↓  Fine-tuning con nuestros datos
     Clasificador  →  NORMAL / PNEUMONIA
```

## Dos fases (VGG16 y ResNet50):
1. **Feature Extraction**: base congelada, entrenar solo clasificador
2. **Fine-Tuning**: descongelar últimas capas, LR muy bajo (1e-5)

**¿Por qué?** Nuestro dataset (~5200 imgs) es pequeño para entrenar desde cero sin overfitting.

> **PneuNet** es la excepción: su arquitectura específica de dominio le permite aprender directamente desde radiografías, **sin Transfer Learning**.

---

# Modelo 3 — PneuNet (Modelo Propio)

## Frontiers in Medicine, 2025 — Diseñado para radiografías de tórax

**Sin Transfer Learning** — entrenado desde cero sobre este dataset

### 4 innovaciones clave:

| Módulo | Innovación |
|--------|-----------|
| **Depthwise Sep. Conv** | 8× menos multiplicaciones que conv estándar |
| **SE Block** | Atención por canal: aprende qué filtros importan |
| **ASPP** | Contexto multi-escala: dilatación 1, 3, 6 |
| **Learnable Pooling** | Atención espacial: enfoca regiones pulmonares |

**~1.84M parámetros** → 75× más liviano que VGG16

---

# Modelo 1 — VGG16

## Simonyan & Zisserman, 2014 (2° ILSVRC)

- **16 capas** con pesos (13 conv + 3 FC)
- Solo filtros **3×3**, todo uniforme
- 5 bloques convolucionales + MaxPooling
- **~138M parámetros**

## Nuestro clasificador:
```
Base VGG16 (congelada / parcial)
  → GlobalAveragePooling2D
  → Dense(512, ReLU) → BN → Dropout(0.5)
  → Dense(256, ReLU) → Dropout(0.3)
  → Dense(1, Sigmoid)
```
*Fine-tuning: bloque 5 (últimas 4 capas)*

---

# Modelo 2 — ResNet50

## He et al., 2015 (1° ILSVRC) — Innovación: Skip Connections

$$H(x) = F(x) + x$$

El gradiente puede fluir directamente →  **sin desvanecimiento**

- **50 capas**, bloques *bottleneck* (1×1 → 3×3 → 1×1)
- Batch Normalization integrada en cada bloque
- **~25M parámetros** (5.5× menos que VGG16!)

## Nuestro clasificador:
```
Base ResNet50 (congelada / parcial)
  → GlobalAveragePooling2D
  → Dense(512, ReLU) → BN → Dropout(0.5)
  → Dense(256, ReLU) → Dropout(0.3)
  → Dense(1, Sigmoid)
```

---

# Estrategia de Entrenamiento

| Configuración       | VGG16/ResNet50 Fase 1 | VGG16/ResNet50 Fase 2 | PneuNet          |
|---------------------|-----------------------|-----------------------|------------------|
| Base convolucional  | **Congelada**         | Parcialmente libre    | N/A (desde cero) |
| Capas entrenables   | Solo clasificador     | Últimas 4-10 capas    | **Todas**        |
| Learning Rate       | **1e-3**              | **1e-5**              | **1e-3**         |
| Max Epochs          | 15                    | 10                    | **50**           |
| Early Stopping      | Paciencia 4           | Paciencia 5           | Paciencia **7**  |

## Callbacks (todos los modelos):
- **EarlyStopping** — evita overfitting
- **ReduceLROnPlateau** — ajuste adaptativo del LR
- **ModelCheckpoint** — guarda el mejor modelo

---

# Data Augmentation

## Solo en entrenamiento — transformaciones clínicamente válidas

| Transformación       | Valor   | ¿Por qué? |
|----------------------|---------|-----------|
| Flip horizontal      | ✓       | Orientación del paciente puede variar |
| Rotación             | ±10°    | Leve inclinación en radiografías |
| Zoom                 | ±10%    | Variaciones de distancia foco-placa |
| Shift H/V            | ±5%     | Posicionamiento del paciente |
| Brillo               | ±10%    | Variación de exposición |

> **No** se aplica flip vertical ni rotaciones mayores — serían imágenes médicamente inválidas.

---

# Curvas de Entrenamiento

![w:1000](curvas_entrenamiento.png)

VGG16/ResNet50: línea punteada = inicio Fine-Tuning | PneuNet: fase única

---

# Matrices de Confusión

![w:1100](matrices_confusion.png)

**Crítico en medicina:** minimizar **Falsos Negativos** (PNEUMONIA predicha como NORMAL)

---

# Curvas ROC

![w:600](curvas_roc.png)

*AUC-ROC cercano a 1.0 indica excelente poder discriminativo en todos los umbrales*

---

# Comparación de Métricas

![w:900](comparacion_metricas.png)

---

# Tabla Comparativa Final

| Métrica        | VGG16      | ResNet50   | PneuNet    |
|----------------|------------|------------|------------|
| Accuracy       | 96.31%     | 93.75%     | *ver CSV*  |
| Precision      | 95.76%     | 98.08%     | *ver CSV*  |
| **Recall**     | **98.46%** | **91.79%** | *ver CSV*  |
| F1-Score       | 97.09%     | 94.83%     | *ver CSV*  |
| AUC-ROC        | 99.24%     | 98.46%     | *ver CSV*  |
| Parámetros     | ~138M      | ~25M       | **~1.84M** |
| Transfer Learn.| Sí         | Sí         | **No**     |

> **Recall = Sensibilidad** — la métrica más importante en diagnóstico médico.

**Lectura rápida:** VGG16 → mayor Recall | ResNet50 → mayor Especificidad | PneuNet → 75× más liviano, sin TL

---

# Análisis Comparativo

## VGG16
✅ Arquitectura simple y bien documentada  
✅ Excelente base para Transfer Learning  
❌ 138M parámetros → lento, alto consumo de memoria  
❌ Sin mecanismos anti-vanishing gradient  

## ResNet50
✅ Skip connections → entrenamiento estable en profundidad  
✅ 5.5× menos parámetros que VGG16  
✅ BatchNorm integrada → convergencia más estable  
❌ Mayor complejidad arquitectónica  

## PneuNet
✅ ~1.84M parámetros → **75× más liviano que VGG16**  
✅ Diseñado específicamente para radiografías de tórax  
✅ ASPP captura estructuras pulmonares multi-escala  
✅ SE block + Learnable Pooling → atención interpretable  
✅ No requiere pesos pre-entrenados (~600 MB descarga)  
❌ Curva de aprendizaje más lenta (entrena desde cero)  

---

# Decisiones Clave Justificadas

1. **Transfer Learning en VGG16/ResNet50:** características de bajo nivel (bordes, texturas) son universales entre dominio fotográfico y médico. Con ~5200 imágenes, entrenar desde cero genera overfitting severo.

2. **Fine-Tuning parcial:** LR altos o muchas capas descongeladas destruyen el conocimiento transferido.

3. **PneuNet sin Transfer Learning:** su arquitectura ASPP+SE fue diseñada para el dominio médico; captura patrones pulmonares sin necesidad de representaciones de fotografías naturales.

4. **GlobalAveragePooling / Learnable Pooling:** reducen parámetros y actúan como regularizadores; Learnable Pooling agrega atención espacial.

5. **Class Weights:** desbalance 1:2.9; sin corrección el modelo ignoraría la clase NORMAL.

6. **Augmentation moderada:** transformaciones físicamente plausibles en radiografías médicas.

---

# Conclusiones

## ¿Qué logramos?

- **Tres modelos** CNN para detección de neumonía, con enfoques distintos
- VGG16 y ResNet50: Transfer Learning desde ImageNet con fine-tuning
- PneuNet: diseño específico de dominio, entrenado desde cero, **75× más liviano**
- Los tres superan la línea base de clasificador aleatorio por un amplio margen

## ¿Qué aprendimos?

- Transfer Learning es **muy eficaz** con datasets médicos pequeños
- Un modelo diseñado para el dominio (**PneuNet**) puede ser competitivo sin TL
- El manejo del **desbalance de clases** impacta directamente en el Recall
- Eficiencia ≠ pérdida de rendimiento: ~1.84M params pueden ser suficientes

---

# Trabajo Futuro

- **Grad-CAM en PneuNet**: visualizar los mapas de atención del SE block para interpretar qué regiones pulmonares activaron la predicción
- **Ensemble**: combinar VGG16 (alto Recall) + ResNet50 (alta Especificidad) + PneuNet (eficiencia)
- **Ajuste del umbral de decisión**: explorar umbrales distintos de 0.5 para maximizar Recall clínico
- **Validación externa**: probar los modelos en datasets de otras poblaciones (NIH ChestX-ray14, CheXpert)

---

# Bibliografía

- Simonyan & Zisserman (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556
- He et al. (2015). *Deep Residual Learning for Image Recognition*. arXiv:1512.03385
- Hu et al. (2018). *Squeeze-and-Excitation Networks*. CVPR 2018. arXiv:1709.01507
- Kermany et al. (2018). *Identifying Medical Diagnoses by Image-Based Deep Learning*. Cell, 172(5)
- PneuNet authors (2025). *PneuNet: a lightweight CNN with multiscale feature fusion for automated pneumonia detection*. Frontiers in Medicine. DOI: 10.3389/fmed.2025.1713587

---

# ¡Gracias!

## Preguntas

**Dataset:** Chest X-Ray Images (Pneumonia) — Kaggle  
**Modelos:** VGG16 + ResNet50 (Transfer Learning) + PneuNet (desde cero)  
**Framework:** TensorFlow/Keras  
**Tarea:** Clasificación binaria NORMAL / PNEUMONIA
