"""
app.py — Interfaz Gradio para detección de neumonía en radiografías de tórax.

Uso:
    pip install gradio
    python app.py
"""

import os
import numpy as np
import gradio as gr
from PIL import Image
from tensorflow import keras

# ── Configuración ──────────────────────────────────────────────────────────────
IMG_SIZE  = (224, 224)
THRESHOLD = 0.5

MODELOS_CFG = {
    'VGG16':    {'path': 'vgg16_best_ft.keras',    'preprocess': 'vgg16'},
    'ResNet50': {'path': 'resnet50_best_ft.keras',  'preprocess': 'resnet50'},
    'PneuNet':  {'path': 'pneunet_best.keras',      'preprocess': 'pneunet'},
}

# ── Caché de modelos cargados ──────────────────────────────────────────────────
_cache = {}

def get_model(nombre):
    if nombre not in _cache:
        path = MODELOS_CFG[nombre]['path']
        if not os.path.exists(path):
            return None
        _cache[nombre] = keras.models.load_model(path)
    return _cache[nombre]


def get_preprocess(nombre):
    tipo = MODELOS_CFG[nombre]['preprocess']
    if tipo == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input
    if tipo == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    return lambda x: x / 255.0


# ── Opciones del dropdown (solo modelos disponibles) ───────────────────────────
def modelos_disponibles():
    disponibles = [n for n, cfg in MODELOS_CFG.items() if os.path.exists(cfg['path'])]
    return disponibles if disponibles else list(MODELOS_CFG.keys())


# ── Función de predicción ──────────────────────────────────────────────────────
def predecir(imagen: Image.Image, nombre_modelo: str):
    if imagen is None:
        return "Cargá una imagen primero.", None, None

    modelo = get_model(nombre_modelo)
    if modelo is None:
        path = MODELOS_CFG[nombre_modelo]['path']
        return f"El modelo '{nombre_modelo}' no está entrenado aún.\nArchivo esperado: {path}", None, None

    preprocess = get_preprocess(nombre_modelo)

    img = imagen.convert('RGB').resize(IMG_SIZE)
    x   = np.array(img, dtype=np.float32)
    x   = np.expand_dims(x, axis=0)
    x   = preprocess(x)

    prob_pneumonia = float(modelo.predict(x, verbose=0)[0][0])
    prob_normal    = 1.0 - prob_pneumonia
    label          = 'PNEUMONIA' if prob_pneumonia >= THRESHOLD else 'NORMAL'

    if label == 'PNEUMONIA':
        resultado = f"🔴 NEUMONÍA DETECTADA\n\nConfianza: {prob_pneumonia:.1%}"
    else:
        resultado = f"🟢 NORMAL\n\nConfianza: {prob_normal:.1%}"

    barras = {
        'NORMAL':    float(prob_normal),
        'PNEUMONIA': float(prob_pneumonia),
    }

    detalle = (
        f"Modelo:          {nombre_modelo}\n"
        f"P(PNEUMONIA):    {prob_pneumonia:.4f}\n"
        f"P(NORMAL):       {prob_normal:.4f}\n"
        f"Umbral:          {THRESHOLD}\n"
        f"Predicción:      {label}"
    )

    return resultado, barras, detalle


# ── UI ─────────────────────────────────────────────────────────────────────────
disponibles = modelos_disponibles()
default_modelo = disponibles[0] if disponibles else 'VGG16'

with gr.Blocks(title="Detector de Neumonía") as demo:
    gr.Markdown(
        """
        # 🫁 Detector de Neumonía en Radiografías de Tórax
        Subí una radiografía de tórax (chest X-ray) y seleccioná el modelo para clasificarla
        como **NORMAL** o **PNEUMONIA**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            imagen_input = gr.Image(
                type='pil',
                label='Radiografía de tórax',
                height=320,
            )
            modelo_input = gr.Dropdown(
                choices=disponibles,
                value=default_modelo,
                label='Modelo',
            )
            boton = gr.Button('Clasificar', variant='primary', size='lg')

            gr.Examples(
                examples=[
                    [os.path.join('dataset', 'chest_xray', 'test', 'PNEUMONIA',
                                  os.listdir(os.path.join('dataset', 'chest_xray', 'test', 'PNEUMONIA'))[0]),
                     default_modelo],
                    [os.path.join('dataset', 'chest_xray', 'test', 'NORMAL',
                                  os.listdir(os.path.join('dataset', 'chest_xray', 'test', 'NORMAL'))[0]),
                     default_modelo],
                ],
                inputs=[imagen_input, modelo_input],
                label='Ejemplos del test set',
            )

        with gr.Column(scale=1):
            resultado_output = gr.Textbox(
                label='Resultado',
                lines=3,
            )
            barras_output = gr.Label(
                label='Probabilidades',
                num_top_classes=2,
            )
            detalle_output = gr.Textbox(
                label='Detalle',
                lines=6,
            )

    boton.click(
        fn=predecir,
        inputs=[imagen_input, modelo_input],
        outputs=[resultado_output, barras_output, detalle_output],
    )

    imagen_input.change(
        fn=predecir,
        inputs=[imagen_input, modelo_input],
        outputs=[resultado_output, barras_output, detalle_output],
    )

    gr.Markdown(
        """
        ---
        **Nota:** Este modelo tiene fines académicos y no debe usarse como herramienta clínica real.
        """
    )

if __name__ == '__main__':
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
