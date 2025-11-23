#!/usr/bin/env python
# coding: utf-8

"""
analize.py
-----------
Script para analizar el dataset de clima y los modelos entrenados.
Genera:
- Distribución de clases
- Imágenes de ejemplo
- Historial de entrenamiento (accuracy y loss)
- Matriz de confusión
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.image import imread
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------------------------------------
# Rutas
# --------------------------------------
train_path = './balanced_train'
test_path = './dataset/test'
model11 = 'saved_model/1/model-11.keras'
model25 = 'saved_model/1/model-25.keras'

# --------------------------------------
# Función para mostrar imágenes de ejemplo
# --------------------------------------
def show_sample_images(dataset_path, save_path='./samples/rainbow1.png', nrows=3, ncols=4):
    fig = plt.figure(figsize=(6,6))
    for i, folder in enumerate(os.listdir(dataset_path)):
        first_img = os.listdir(os.path.join(dataset_path, folder))[0]
        img = imread(os.path.join(dataset_path, folder, first_img))
        fig.add_subplot(nrows, ncols, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(folder)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Imagen de ejemplo guardada en {save_path}")

# --------------------------------------
# Función para analizar la distribución de clases
# --------------------------------------
def plot_class_distribution(dataset_path, save_path='class_distribution.png'):
    images_count = {folder: len(os.listdir(os.path.join(dataset_path, folder))) for folder in os.listdir(dataset_path)}
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(images_count.values()), y=list(images_count.keys()), palette='crest', orient='h')
    plt.xlabel('Número de imágenes')
    plt.ylabel('Clases')
    plt.title('Distribución de clases')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Gráfico de distribución de clases guardado en {save_path}")

# --------------------------------------
# Función para evaluar modelo
# --------------------------------------
def evaluate_model(model_path, test_path, batch_size=64, image_size=(224,224)):
    print(f"[INFO] Evaluando modelo: {model_path}")
    model = load_model(model_path)
    test_gen = ImageDataGenerator(rescale=1/255)
    test_ds = test_gen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluación
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    # Predicciones
    y_pred_prob = model.predict(test_ds)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_ds.classes
    
    # Reporte
    print(classification_report(y_true, y_pred, target_names=list(test_ds.class_indices.keys())))
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='mako')
    ax.set_xticklabels(test_ds.class_indices.keys(), rotation=45)
    ax.set_yticklabels(test_ds.class_indices.keys(), rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix {os.path.basename(model_path)}')
    cm_file = f'cm_{os.path.basename(model_path)}.png'
    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()
    print(f"[INFO] Matriz de confusión guardada en {cm_file}")
    
# --------------------------------------
# Función para mostrar historial del entrenamiento
# --------------------------------------
def plot_training_history(history, save_path='training_history.png'):
    metrics = pd.DataFrame(history.history)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(metrics['accuracy'], label='Train Acc')
    plt.plot(metrics['val_accuracy'], label='Val Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Epochs')
    
    plt.subplot(1,2,2)
    plt.plot(metrics['loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epochs')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Historial de entrenamiento guardado en {save_path}")

# --------------------------------------
# Ejecutar funciones
# --------------------------------------
if __name__ == "__main__":
    show_sample_images(train_path)
    plot_class_distribution(train_path)
    
    # Modelos previamente entrenados
    # for model_path in [model25, model11]:
    #     evaluate_model(model_path, test_path)
    
    print("[INFO] Análisis completo.")
