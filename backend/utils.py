# """
# Utility functions for deepfake detection
# Includes preprocessing, Grad-CAM visualization, and helper functions
# """

# import numpy as np
# import cv2
# from tensorflow import keras
# import tensorflow as tf
# from PIL import Image
# import io

# IMG_SIZE = 224

# def preprocess_image(image_bytes):
#     """
#     Preprocess uploaded image for model prediction
    
#     Args:
#         image_bytes: Raw image bytes from upload
    
#     Returns:
#         Preprocessed numpy array ready for model
#     """
#     # Convert bytes to PIL Image
#     image = Image.open(io.BytesIO(image_bytes))
    
#     # Convert to RGB if needed
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
    
#     # Resize to model input size
#     image = image.resize((IMG_SIZE, IMG_SIZE))
    
#     # Convert to numpy array
#     img_array = np.array(image)
    
#     # Normalize pixel values to [0, 1]
#     img_array = img_array.astype('float32') / 255.0
    
#     # Add batch dimension
#     img_array = np.expand_dims(img_array, axis=0)
    
#     return img_array

# def detect_face(image_bytes):
#     """
#     Optional: Detect and crop face from image using OpenCV
#     Falls back to original image if no face detected
    
#     Args:
#         image_bytes: Raw image bytes
    
#     Returns:
#         Cropped face image bytes or original if no face found
#     """
#     try:
#         # Convert bytes to numpy array
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         # Load face cascade classifier
#         face_cascade = cv2.CascadeClassifier(
#             cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
#         )
        
#         # Detect faces
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
#         # If face found, crop it
#         if len(faces) > 0:
#             (x, y, w, h) = faces[0]  # Take first face
#             # Add margin
#             margin = int(0.2 * w)
#             x = max(0, x - margin)
#             y = max(0, y - margin)
#             w = w + 2 * margin
#             h = h + 2 * margin
            
#             face_img = img[y:y+h, x:x+w]
            
#             # Convert back to bytes
#             _, buffer = cv2.imencode('.jpg', face_img)
#             return buffer.tobytes()
        
#         return image_bytes
    
#     except Exception as e:
#         print(f"Face detection error: {e}")
#         return image_bytes

# def generate_gradcam(model, img_array, layer_name='block14_sepconv2_act'):
#     """
#     Generate Grad-CAM heatmap for model explainability
#     Shows which regions of the image influenced the prediction
    
#     Args:
#         model: Trained Keras model
#         img_array: Preprocessed image array
#         layer_name: Name of the layer to visualize
    
#     Returns:
#         Heatmap overlay on original image
#     """
#     try:
#         # Create a model that maps input to target layer output and final prediction
#         grad_model = keras.Model(
#             inputs=model.input,
#             outputs=[model.get_layer(layer_name).output, model.output]
#         )
        
#         # Compute gradient of prediction with respect to target layer
#         with tf.GradientTape() as tape:
#             conv_outputs, predictions = grad_model(img_array)
#             loss = predictions[:, 0]
        
#         # Get gradients
#         grads = tape.gradient(loss, conv_outputs)
        
#         # Pool gradients across spatial dimensions
#         pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
#         # Weight feature maps by gradients
#         conv_outputs = conv_outputs[0]
#         heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
#         heatmap = tf.squeeze(heatmap)
        
#         # Normalize heatmap
#         heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
#         heatmap = heatmap.numpy()
        
#         # Resize heatmap to image size
#         heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        
#         # Convert to RGB
#         heatmap = np.uint8(255 * heatmap)
#         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
#         # Get original image
#         original_img = np.uint8(255 * img_array[0])
        
#         # Superimpose heatmap on original image
#         superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
#         # Convert to bytes
#         _, buffer = cv2.imencode('.jpg', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        
#         return buffer.tobytes()
    
#     except Exception as e:
#         print(f"Grad-CAM error: {e}")
#         return None

# def get_prediction_label(confidence):
#     """
#     Convert model confidence to human-readable label
    
#     Args:
#         confidence: Model output (0-1, where 1 = fake)
    
#     Returns:
#         tuple: (label, confidence_percentage, class)
#     """
#     if confidence > 0.5:
#         label = "FAKE"
#         percentage = confidence * 100
#         pred_class = "fake"
#     else:
#         label = "REAL"
#         percentage = (1 - confidence) * 100
#         pred_class = "real"
    
#     return label, percentage, pred_class





"""
Utility functions for deepfake detection
Includes preprocessing, Grad-CAM visualization, and helper functions
"""

import numpy as np
import cv2
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import io

IMG_SIZE = 224

def preprocess_image(image_bytes):
    """
    Preprocess uploaded image for model prediction
    """
    image = Image.open(io.BytesIO(image_bytes))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def detect_face(image_bytes):
    """
    Detect and crop face from image using OpenCV (optional)
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            margin = int(0.2 * w)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = w + 2 * margin
            h = h + 2 * margin

            face_img = img[y:y+h, x:x+w]
            _, buffer = cv2.imencode('.jpg', face_img)
            return buffer.tobytes()

        return image_bytes

    except Exception as e:
        print(f"Face detection error: {e}")
        return image_bytes


def generate_gradcam(model, img_array, layer_name=None):
    """
    Generate Grad-CAM heatmap for explainability
    """
    try:
        if layer_name is None:
            # Try to find last conv layer automatically
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    layer_name = layer.name
                    break

        grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # For binary classification, pick the FAKE class (usually index 0 or 1)
            if predictions.shape[-1] == 1:
                loss = predictions[:, 0]
            else:
                loss = predictions[:, 1]  # assuming index 1 = fake

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        original_img = np.uint8(255 * img_array[0])
        superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
        return buffer.tobytes()

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def get_prediction_label(prediction):
    """
    Convert model output to readable label.
    Handles both binary (1 neuron) and softmax (2 neurons) outputs.
    """
    try:
        if isinstance(prediction, np.ndarray):
            prediction = prediction.flatten()

        if len(prediction) == 1:
            # Sigmoid output (0 = real, 1 = fake)
            confidence = float(prediction[0])
        elif len(prediction) == 2:
            # Softmax output (index 0 = real, index 1 = fake)
            confidence = float(prediction[1])
        else:
            confidence = 0.5  # default fallback

        if confidence >= 0.5:
            label = "FAKE"
            percentage = confidence * 100
            pred_class = "fake"
        else:
            label = "REAL"
            percentage = (1 - confidence) * 100
            pred_class = "real"

        return label, round(percentage, 2), pred_class

    except Exception as e:
        print(f"Prediction label error: {e}")
        return "UNKNOWN", 0.0, "unknown"
