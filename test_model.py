from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("trained_plant_disease_model.keras")

img = image.load_img("AppleCedarRust1.JPG", target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
print(np.argmax(pred))
print(pred)