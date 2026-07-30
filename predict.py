import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model("grain_classifier.keras")
model.summary()

# Same class order as your training run
class_names = ['chaana_dal', 'chole', 'harbara', 'masur_dal', 'matki',
               'moong', 'peanut', 'rice', 'tur_dal', 'wheat']

def predict_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # add batch dimension

    predictions = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    print(f"Predicted: {predicted_class} ({confidence*100:.2f}% confidence)")
    return predicted_class, confidence

if __name__ == "__main__":
 predict_image("test_images/6_harbhara_003.jpg")