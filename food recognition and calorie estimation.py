import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define paths
dataset_path = '/path/to/food-101/images'

# Create data generators
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(101, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)


# Example: Replace with accurate data
calorie_map = {
    "apple_pie": 296,
    "bibimbap": 560,
    "caesar_salad": 150,
   
}


def estimate_calories(image, model, class_indices, calorie_map):
    img = tf.keras.preprocessing.image.load_img(image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_label = list(class_indices.keys())[list(class_indices.values()).index(class_idx)]
    estimated_calories = calorie_map.get(class_label, "Unknown")

    return class_label, estimated_calories

img_path = '/path/to/sample_food.jpg'
label, calories = estimate_calories(img_path, model, train_generator.class_indices, calorie_map)

print(f"Predicted Food: {label}, Estimated Calories: {calories} kcal")


model.save('food_calorie_model.h5')

# model = tf.keras.models.load_model('food_calorie_model.h5') # this is optional

