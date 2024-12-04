import os
import cv2
import numpy as np
import random

def alter_color(image):
    # lterazione dei colori dell'immagine.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = random.randint(30, 70) 
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def blur_image(image):
    # Sfocatura dell'immagine.
    ksize = random.choice([(15, 15), (25, 25), (35, 35)])  
    sigmaX = random.uniform(0, 5)  
    return cv2.GaussianBlur(image, ksize, sigmaX)

def double_vision_effect(image):
    # Crea un effetto di doppia visione traslando l'immagine.
    shift = random.randint(3, 7)  
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, shift], [0, 1, 0]])  
    shifted = cv2.warpAffine(image, M, (cols, rows))
    return cv2.addWeighted(image, 0.5, shifted, 0.5, 0)

def apply_alcohol_effect(image, output_folder, filename):
    # Applica e salva l'effetto di alterazione dei colori
    color_altered = alter_color(image)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_color_altered.jpg"), color_altered)

    # Applica e salva l'effetto di sfocatura
    blurred = blur_image(color_altered)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_blurred.jpg"), blurred)

    # Applica e salva l'effetto di doppia visione
    alcohol_effect = double_vision_effect(blurred)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_double_vision.jpg"), alcohol_effect)

    return alcohol_effect

def process_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Leggi tutte le immagini dalla cartella
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # Processa e salva le immagini con il filtro Alcol
    for filename in images:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Applica gli effetti e salva le immagini alterate
        apply_alcohol_effect(image, output_folder, os.path.splitext(filename)[0])

# Cartelle root
process_folder('input_img_flat', 'output_img_distorted')