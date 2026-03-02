
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

async def KerasModel():
    r"""Initializes and trains the Keras model if it is not already available. 
    This function checks if the Keras model is already initialized; 
    if not, it attempts to initialize and train the model using the WordDefinitionModel class. 
    If successful, it returns the trained Keras model; otherwise, it returns None."""

    keras_model = None

    if keras_model is None:
        print("❌ Keras model not available.")
        if await wordDefinitionModel.initializeAndTrainKerasModel():
            if wordDefinitionModel.kerasModel is not None:
                keras_model = wordDefinitionModel.kerasModel
                print("✅ Keras model initialized and trained successfully.")
                return keras_model
        else:
            print("❌ Failed to initialize and train the Keras model.")
            return None


async def main():
    kerasModel = await KerasModel()
    if kerasModel is not None:
        predicts = kerasModel.getPredictions("It covers 71% of the Earth's surface")
        print(f"Predictions: {predicts}")


asyncio.run(main())
