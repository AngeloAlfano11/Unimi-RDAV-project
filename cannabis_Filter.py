import os
import cv2
import numpy as np
import random

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



def apply_cannabis_effect(image, output_folder, filename):
    # Applica l'effetto cannabis all'immagine.
    high_contrast = increase_contrast(image)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_high_contrast.jpg"), high_contrast)
    altered_vision = alter_stereoscopic_vision(high_contrast)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_altered_vision.jpg"), altered_vision)
    light_sensitive = increase_light_sensitivity(altered_vision)
    cv2.imwrite(os.path.join(output_folder, f"{filename}_light_sensitive.jpg"), light_sensitive)
    return light_sensitive

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
        apply_cannabis_effect(image, output_folder, os.path.splitext(filename)[0])

# Cartelle root
process_folder('input_img_flat', 'output_img_distorted')