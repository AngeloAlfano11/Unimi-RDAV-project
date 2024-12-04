import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Preparazione del Dataset

dataset_dir = "Dataset"  # Percorso relativo alla cartella Dataset

# Creazione di generatori per training e validazione
datagen = ImageDataGenerator(
    rescale=1.0 / 255,        
    validation_split=0.2      
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    batch_size=32,            
    class_mode='sparse',      
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# 2. Definizione della Rete Neurale

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 3. Addestramento del Modello

model.fit(
    train_generator,
    epochs=3,
    validation_data=validation_generator
)


# 4. Valutazione del Modello

test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {test_accuracy:.2f}")


# 5. Classificazione delle Immagini da una Cartella

def classify_images_from_folder(folder_path, model, class_names):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, filename)
            image = load_img(image_path, target_size=(256, 256))
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            prediction = model.predict(image_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            
            plt.imshow(image)
            plt.title(f"Predicted Class: {class_names[predicted_class]}")
            plt.axis('off')
            plt.show()

# Percorso della cartella contenente le immagini da classificare
test_images_folder = "output_img_distorted" # Percorso relativo alla cartella output

# Classifica le immagini dalla cartella
classify_images_from_folder(test_images_folder, model, list(train_generator.class_indices.keys()))
