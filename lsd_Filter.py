import os
import cv2
import numpy as np
import random

def distort_shapes_and_colors(image):
    # Distorsioni di forme e colori.
    rows, cols, ch = image.shape
    distortion = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            offset_x = int(random.uniform(2.0, 4.0) * np.sin(2 * np.pi * i / 30.0))
            offset_y = int(random.uniform(2.0, 4.0) * np.cos(2 * np.pi * j / 30.0))
            if i + offset_x < rows and j + offset_y < cols:
                distortion[i, j] = image[(i + offset_x) % rows, (j + offset_y) % cols]
            else:
                distortion[i, j] = image[i, j]
    return distortion

def add_light_halos(image):
    # Aggiunge aloni di luce.
    alpha = random.uniform(0.30, 0.50)  # Randomizza il valore di alpha tra 0.85 e 0.95
    beta = 1 - alpha
    return cv2.addWeighted(image, alpha, cv2.GaussianBlur(image, (15, 15), 5), beta, 0)

def enhance_colors(image):
    # Colori intensi e brillanti.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_increase = random.randint(20, 30)  # Randomizza l'aumento della saturazione tra 20 e 30
    hsv[..., 1] = cv2.add(hsv[..., 1], saturation_increase)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_lsd_effect(image, output_folder, filename):
    #Applica l'effetto LSD all'immagine e salva ogni passaggio.
    distorted_image = distort_shapes_and_colors(image)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_distorted.jpg"), distorted_image)
    light_halos_image = add_light_halos(distorted_image)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_light_halos.jpg"), light_halos_image)
    enhanced_colors_image = enhance_colors(light_halos_image)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_enhanced_colors.jpg"), enhanced_colors_image)
    return enhanced_colors_image

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
        apply_lsd_effect(image, output_folder, os.path.splitext(filename)[0])

# Cartelle root
process_folder('input_img_flat', 'output_img_distorted')