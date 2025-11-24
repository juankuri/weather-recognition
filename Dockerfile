# Usa imagen oficial de TensorFlow
FROM tensorflow/tensorflow:latest

# Crea carpeta para el modelo
RUN mkdir -p /models/weather_model/1

# Copia tu modelo al contenedor
COPY saved_model/2/model-06.keras /models/weather_model/1/

# Exporta el modelo a formato SavedModel para TF Serving
RUN python -c "import tensorflow as tf; model = tf.keras.models.load_model('/models/weather_model/1/model-06.keras'); model.save('/models/weather_model/1', save_format='tf')"

# Expone el puerto para TF Serving
EXPOSE 8501

# Comando para correr TF Serving
ENTRYPOINT ["tensorflow_model_server", "--rest_api_port=8501", "--model_name=weather_model", "--model_base_path=/models/weather_model"]
