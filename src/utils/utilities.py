
import os

import numpy as np
import pandas as pd
import tensorflow as tf

import pickle



def save_checkpoint(data: np.ndarray | list, filename: str):
    r"""Saves data to a checkpoint file using pickle. This allows for resuming training or reusing preprocessed data without having to redo expensive computations.
    Args:
        data: The data to save (e.g., lists of augmented definitions, embeddings).
        filename (str): The name of the checkpoint file (e.g., 'augmented_definitions.pkl').
    """
    print(f"💾 Saving checkpoint: {filename}...")
    with open(f"checkpoints/{filename}", "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(filename: str) -> np.ndarray | list:
    r"""Loads data from a checkpoint file if it exists. This is useful for resuming training or reusing preprocessed data.
    Args:
        filename (str): The name of the checkpoint file (e.g., 'augmented_definitions.pkl').
    Returns:
        The data loaded from the checkpoint file, or an empty list if the file does not exist.
    """
    if os.path.exists(f"checkpoints/{filename}"):
        with open(f"checkpoints/{filename}", "rb") as f:
            return pickle.load(f)
    return []


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