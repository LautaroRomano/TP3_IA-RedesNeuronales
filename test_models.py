"""
test_models.py — Evaluación de modelos entrenados sobre el test set o una imagen individual.

Uso:
    # Evaluar todos los modelos disponibles sobre el test set completo:
    python test_models.py

    # Evaluar solo un modelo específico:
    python test_models.py --modelo vgg16

    # Clasificar una imagen individual:
    python test_models.py --imagen ruta/a/imagen.jpg

    # Clasificar una imagen con un modelo específico:
    python test_models.py --imagen ruta/a/imagen.jpg --modelo resnet50
"""

import os
import sys
import argparse
import numpy as np

# ── Configuración ──────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
THRESHOLD  = 0.5
TEST_DIR   = os.path.join('dataset', 'chest_xray', 'test')

MODELOS = {
    'vgg16': {
        'path': 'vgg16_best_ft.keras',
        'preprocess': 'vgg16',
    },
    'resnet50': {
        'path': 'resnet50_best_ft.keras',
        'preprocess': 'resnet50',
    },
    'pneunet': {
        'path': 'pneunet_best.keras',
        'preprocess': 'pneunet',
    },
}


def get_preprocess_fn(nombre):
    if nombre == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
    if nombre == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    if nombre == 'pneunet':
        return lambda x: x / 255.0
    raise ValueError(f'Preprocesamiento desconocido: {nombre}')


def cargar_modelo(nombre):
    from tensorflow import keras
    cfg = MODELOS[nombre]
    if not os.path.exists(cfg['path']):
        print(f'  [!] {cfg["path"]} no encontrado — salteando.')
        return None, None
    print(f'  Cargando {cfg["path"]} ...')
    modelo = keras.models.load_model(cfg['path'])
    preprocess = get_preprocess_fn(cfg['preprocess'])
    return modelo, preprocess


# ── Evaluación sobre el test set ───────────────────────────────────────────────

def evaluar_test(nombres_modelos):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_curve, auc, confusion_matrix,
    )
    import pandas as pd

    if not os.path.isdir(TEST_DIR):
        sys.exit(f'Error: no se encontró el directorio de test en "{TEST_DIR}".')

    filas = []

    for nombre in nombres_modelos:
        print(f'\n[{nombre.upper()}]')
        modelo, preprocess = cargar_modelo(nombre)
        if modelo is None:
            continue

        datagen = ImageDataGenerator(preprocessing_function=preprocess)
        test_gen = datagen.flow_from_directory(
            TEST_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False,
        )

        print('  Prediciendo sobre el test set ...')
        probs  = modelo.predict(test_gen, verbose=0).ravel()
        preds  = (probs >= THRESHOLD).astype(int)
        labels = test_gen.classes

        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        filas.append({
            'Modelo':         nombre.upper(),
            'Accuracy':       round(accuracy_score(labels, preds)   * 100, 2),
            'Precision':      round(precision_score(labels, preds)  * 100, 2),
            'Recall':         round(recall_score(labels, preds)     * 100, 2),
            'F1-Score':       round(f1_score(labels, preds)         * 100, 2),
            'AUC-ROC':        round(auc_val                         * 100, 2),
            'Especificidad':  round(especificidad                   * 100, 2),
            'Parametros (M)': round(modelo.count_params() / 1e6, 2),
        })

        print(f'  Accuracy:      {filas[-1]["Accuracy"]}%')
        print(f'  Recall:        {filas[-1]["Recall"]}%')
        print(f'  F1-Score:      {filas[-1]["F1-Score"]}%')
        print(f'  AUC-ROC:       {filas[-1]["AUC-ROC"]}%')
        print(f'  Especificidad: {filas[-1]["Especificidad"]}%')

    if not filas:
        print('\nNo se encontró ningún modelo entrenado.')
        return

    df = pd.DataFrame(filas).set_index('Modelo')
    print('\n' + '=' * 65)
    print('  TABLA COMPARATIVA — TEST SET')
    print('=' * 65)
    print(df.to_string())
    print('=' * 65)

    mejor_recall = df['Recall'].idxmax()
    mejor_f1     = df['F1-Score'].idxmax()
    print(f'\nMejor Recall (diagnóstico médico): {mejor_recall}  ({df.loc[mejor_recall, "Recall"]}%)')
    print(f'Mejor F1-Score:                    {mejor_f1}  ({df.loc[mejor_f1, "F1-Score"]}%)')


# ── Clasificación de una imagen individual ─────────────────────────────────────

def clasificar_imagen(ruta_imagen, nombre_modelo):
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image as kimage

    if not os.path.exists(ruta_imagen):
        sys.exit(f'Error: no se encontró la imagen en "{ruta_imagen}".')

    modelo, preprocess = cargar_modelo(nombre_modelo)
    if modelo is None:
        sys.exit(f'No se pudo cargar el modelo "{nombre_modelo}".')

    img = kimage.load_img(ruta_imagen, target_size=IMG_SIZE)
    x   = kimage.img_to_array(img)
    x   = np.expand_dims(x, axis=0)
    x   = preprocess(x)

    prob  = float(modelo.predict(x, verbose=0)[0][0])
    label = 'PNEUMONIA' if prob >= THRESHOLD else 'NORMAL'
    confianza = prob if label == 'PNEUMONIA' else 1 - prob

    print(f'\nImagen:   {ruta_imagen}')
    print(f'Modelo:   {nombre_modelo.upper()} ({MODELOS[nombre_modelo]["path"]})')
    print(f'Resultado: {label}  (confianza {confianza:.1%})')
    print(f'  P(PNEUMONIA) = {prob:.4f}')
    print(f'  P(NORMAL)    = {1 - prob:.4f}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evalúa los modelos entrenados del TP3.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--modelo', '-m',
        choices=list(MODELOS.keys()),
        default=None,
        help='Modelo a usar (vgg16 | resnet50 | pneunet). Por defecto: todos los disponibles.',
    )
    parser.add_argument(
        '--imagen', '-i',
        default=None,
        metavar='RUTA',
        help='Ruta a una imagen individual para clasificar.',
    )
    args = parser.parse_args()

    if args.imagen:
        modelo = args.modelo or 'vgg16'
        clasificar_imagen(args.imagen, modelo)
    else:
        modelos = [args.modelo] if args.modelo else list(MODELOS.keys())
        evaluar_test(modelos)


if __name__ == '__main__':
    main()
