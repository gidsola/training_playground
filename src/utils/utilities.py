
import tensorflow as tf


def convert_keras_to_tflite(keras_model_path: str, output_file_path: str) -> bool:
    r"""Converts the trained Keras model to TensorFlow Lite format and saves it. This allows for deployment on edge devices where resources are limited. The function checks for the existence of the Keras model, performs the conversion, and handles any exceptions that may occur during the process.
    Returns:
        bool: True if the conversion was successful, False otherwise.
    """
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(keras_model_path)
        tflite_model = converter.convert()
        
        with open(output_file_path, 'wb') as f:
            f.write(tflite_model)

        print(f"💾 TFLite model saved to {output_file_path}.")
        return True
    
    except Exception as e:
        print(f"❌ Error occurred while converting Keras model to TFLite: {e}")
        return False
    