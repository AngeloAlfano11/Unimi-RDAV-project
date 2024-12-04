import os
import cv2
import numpy as np
import random


# Funzioni per l'effetto LSD
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
    alpha = random.uniform(0.30, 0.50)
    beta = 1 - alpha
    return cv2.addWeighted(image, alpha, cv2.GaussianBlur(image, (15, 15), 5), beta, 0)

def enhance_colors(image):
    # Colori intensi e brillanti.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_increase = random.randint(20, 30) 
    hsv[..., 1] = cv2.add(hsv[..., 1], saturation_increase)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_lsd_effect(image):
    # Applica l'effetto LSD all'immagine.
    distorted = distort_shapes_and_colors(image)
    halos = add_light_halos(distorted)
    enhanced_colors = enhance_colors(halos)
    return enhanced_colors

# --------------------------------------------

# Funzioni per l'effetto cannabis
def increase_contrast(image):
    # Aumenta il contrasto dell'immagine.
    alpha = random.uniform(1.2, 1.5)  
    beta = random.randint(20, 40)     
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def alter_stereoscopic_vision(image):
    # Crea un effetto di visione stereoscopica alterata.
    shift = random.randint(10, 20)  # Aumenta il valore di shift tra 10 e 20
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, shift], [0, 1, 0]]) 
    shifted = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    b, g, r = cv2.split(image)
    b_shifted, g_shifted, r_shifted = cv2.split(shifted)
    combined = cv2.merge((b_shifted, g, r_shifted))

    return combined

def increase_light_sensitivity(image):
    # Simula la sensibilità alla luce aumentando la luminosità senza stravolgere i colori, aggiungendo un effetto di bagliore delicato.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_boost = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    l_boost = np.clip(l_boost * random.uniform(1.1, 1.3), 0, 255).astype(np.uint8)
    enhanced_lab = cv2.merge([l_boost, a, b])
    brightened_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    glow_intensity = random.uniform(0.2, 0.4)
    glow = cv2.GaussianBlur(brightened_image, (25, 25), 0)
    result = cv2.addWeighted(brightened_image, 1 - glow_intensity, glow, glow_intensity, 0)
    return result

def apply_cannabis_effect(image):
    # Applica l'effetto cannabis all'immagine.
    high_contrast = increase_contrast(image)
    altered_vision = alter_stereoscopic_vision(high_contrast)
    light_sensitive = increase_light_sensitivity(altered_vision)
    return light_sensitive

# --------------------------------------------

# Funzioni per l'effetto alcol
def alter_color(image):
    # Alterazione dei colori dell'immagine.
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
    shifted = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(image, 0.5, shifted, 0.5, 0)

def apply_alcohol_effect(image):
    # Applica l'effetto alcol all'immagine.
    color_altered = alter_color(image)
    blurred = blur_image(color_altered)
    alcohol_effect = double_vision_effect(blurred)
    return alcohol_effect

# --------------------------------------------

# Funzioni per la creazione del Dataset

import os
import cv2

def process_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Leggi tutte le immagini dalla cartella
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    total_images = len(images)
    third = total_images // 3

    # Suddividi le immagini in tre gruppi
    alcohol_images = images[:third]
    cannabis_images = images[third:2*third]
    lsd_images = images[2*third:]

    # Cartelle di output per ciascun effetto
    alcohol_folder = os.path.join(output_folder, 'Alcol')
    cannabis_folder = os.path.join(output_folder, 'Cannabis')
    lsd_folder = os.path.join(output_folder, 'LSD')

    # Creazione delle cartelle di output se non esistono
    os.makedirs(alcohol_folder, exist_ok=True)
    os.makedirs(cannabis_folder, exist_ok=True)
    os.makedirs(lsd_folder, exist_ok=True)

    # Processa e salva le immagini con il filtro Alcol
    for filename in alcohol_images:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            alcol_effect = apply_alcohol_effect(image)
            base_filename = os.path.basename(image_path)
            name, ext = os.path.splitext(base_filename)
            cv2.imwrite(os.path.join(alcohol_folder, f"{name}_alcol{ext}"), alcol_effect)

    # Processa e salva le immagini con il filtro Cannabis
    for filename in cannabis_images:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            cannabis_effect = apply_cannabis_effect(image)
            base_filename = os.path.basename(image_path)
            name, ext = os.path.splitext(base_filename)
            cv2.imwrite(os.path.join(cannabis_folder, f"{name}_cannabis{ext}"), cannabis_effect)

    # Processa e salva le immagini con il filtro LSD
    for filename in lsd_images:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            lsd_effect = apply_lsd_effect(image)
            base_filename = os.path.basename(image_path)
            name, ext = os.path.splitext(base_filename)
            cv2.imwrite(os.path.join(lsd_folder, f"{name}_lsd{ext}"), lsd_effect)

# Percorso della cartella contenente le immagini da alterare
folder_path = 'flat_img'
output_folder = 'Dataset'

process_folder(folder_path, output_folder)
