
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

import asyncio

from src.models.WordDefinitionModel import WordDefinitionModel

wordDefinitionModel = WordDefinitionModel(
    batch_size=512,
    epochs=30
)

async def prediction_Keras(input: str):
    r"""Generates a prediction using the Keras model for a given input string. The function determines whether to predict a word or a definition based on the input length.
    Args:
        input (str): The input string for which to generate a prediction.
    Returns:
        str: The predicted word or definition.
    """

    keras_model = wordDefinitionModel.kerasModel

    if keras_model is None:
        print("❌ Keras model not available.")
        await wordDefinitionModel.initializeAndTrainKerasModel()
        print("✅ Keras model is ready for predictions.")


predicts = prediction_Keras("liquid covering most of the earths surface")
print(f"Predictions: {predicts}")
