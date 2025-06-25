# STEP 1: Upload ZIP
from google.colab import files
uploaded = files.upload()  # Uploading 'food_images.zip`

# STEP 2: Unzip and rename the folder
import zipfile
import os

with zipfile.ZipFile("prodigy_food_images.zip", 'r') as zip_ref:
    zip_ref.extractall("/content")

# Rename to remove spaces
os.rename("/content/prodigy infotech food image", "/content/food_dataset")

# STEP 3: ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    "/content/food_dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    "/content/food_dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# STEP 4: CNN Model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# STEP 5: Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# STEP 6: Calorie Dictionary
calorie_dict = {
    'biryani_image': 290,   # per 100g
    'pizza_image': 266,
    'burger_image': 295
}

# STEP 7: Predict + Calorie Function
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image_with_calories(img_path, portion_grams=200):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    
    class_name = list(train_generator.class_indices.keys())[class_idx]
    calories_per_100g = calorie_dict.get(class_name, "Unknown")

    print(f"\n🍽️ Predicted Food: {class_name.replace('_image', '').capitalize()}")
    print(f"🔥 Calories per 100g: {calories_per_100g} kcal")

    if calories_per_100g != "Unknown":
        total_cal = (calories_per_100g / 100) * portion_grams
        print(f"⚖️ Estimated Calories for {portion_grams}g: {total_cal:.2f} kcal")
    else:
        print("⚠️ Calorie info not available.")

# STEP 8: Upload a Test Image and Predict
uploaded = files.upload()  # Upload e.g., test_biryani.jpg

predict_image_with_calories("test_biryani.jpg", portion_grams=200)
