import tensorflow as tf
import numpy as np
from PIL import Image

CLASSES = ["dew", "fogsmog", "frost", "glaze", "hail",
           "lightning", "rain", "rainbow", "rime", "sandstorm", "snow"]

model_version = "15"
model = tf.keras.models.load_model(f"saved_model/1/model-{model_version}.keras")
# model = tf.keras.models.load_model("saved_model/1/model-8.keras") 
print(f"Modelo {model_version} cargado correctamente.") 

def predict(path):
    img = Image.open(path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, 0)

    pred = model.predict(img)[0]
    # Ordenar por probabilidad descendente
    pred_dict = dict(zip(CLASSES, pred))
    pred_sorted = dict(sorted(pred_dict.items(), key=lambda item: item[1], reverse=True))

    print("Predicciones:")
    for cls, prob in pred_sorted.items():
        print(f"{cls:10s}: {prob:.2%}")

    idx = np.argmax(pred)
    print(f"\nClase predicha: {CLASSES[idx]}")
    return CLASSES[idx]

print(predict("./samples/snow1.png"))
