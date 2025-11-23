import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

train_path = '/home/kuri/Documents/uni/ia/weather/dataset/train'

# Recuento de imágenes por clase
images_count_dict = {folder: len(os.listdir(os.path.join(train_path, folder))) for folder in os.listdir(train_path)}
max_count = max(images_count_dict.values())  # número de imágenes de la clase más grande

# Configuración de aumento de datos
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Carpeta donde guardar imágenes balanceadas
balanced_path = './balanced_train' 
os.makedirs(balanced_path, exist_ok=True)

for cls in os.listdir(train_path):
    cls_path = os.path.join(train_path, cls)
    balanced_cls_path = os.path.join(balanced_path, cls)
    os.makedirs(balanced_cls_path, exist_ok=True)
    
    current_count = len(os.listdir(cls_path))
    # Copiar imágenes originales
    for img_file in os.listdir(cls_path):
        src = os.path.join(cls_path, img_file)
        dst = os.path.join(balanced_cls_path, img_file)
        if not os.path.exists(dst):
            from shutil import copy
            copy(src, dst)
    
    # Generar imágenes adicionales si la clase es menor que la mayor
    n_to_generate = max_count - current_count
    if n_to_generate > 0:
        images = [img_to_array(load_img(os.path.join(cls_path, f))) for f in os.listdir(cls_path)]
        idx = 0
        for i in range(n_to_generate):
            img = images[idx % len(images)]
            img = img.reshape((1, *img.shape))
            for batch in datagen.flow(img, batch_size=1, save_to_dir=balanced_cls_path, save_prefix='aug', save_format='jpg'):
                break  # generar 1 imagen por iteración
            idx += 1
