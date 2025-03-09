import io
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = 'best_model.keras'
INPUT_SHAPE = (256, 256)
NUM_CLASSES = 8

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image, target_size=INPUT_SHAPE):
    """
    Convertit l'image en RGB, la redimensionne et la normalise.
    Retourne également la taille originale de l'image.
    """
    original_size = image.size  # format (largeur, hauteur)
    image = image.convert('RGB')
    image_resized = image.resize(target_size)
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, original_size

def postprocess_mask(prediction, original_size):
    """
    Transforme la prédiction en masque de segmentation :
    - Applique argmax pour obtenir l'indice de classe par pixel
    - Convertit en image en niveaux de gris et redimensionne au format original
    """
    pred_mask = np.argmax(prediction, axis=-1)[0]
    mask_scaled = (pred_mask * (255 / (NUM_CLASSES - 1))).astype(np.uint8)
    mask_image = Image.fromarray(mask_scaled, mode='L')
    mask_image = mask_image.resize(original_size)
    return mask_image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni.'}), 400
    file = request.files['file']
    try:
        image = Image.open(file.stream)
    except Exception as e:
        return jsonify({'error': 'Fichier image invalide.'}), 400

    image_array, original_size = preprocess_image(image, target_size=INPUT_SHAPE)
    prediction = model.predict(image_array)
    mask_image = postprocess_mask(prediction, original_size)

    img_io = io.BytesIO()
    mask_image.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
