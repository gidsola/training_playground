
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

import asyncio

from src.models.WordDefinitionModel import WordDefinitionModel



async def KerasModel():
    r"""Initializes and trains the Keras model if it is not already available. 
    This function checks if the Keras model is already initialized; 
    if not, it attempts to initialize and train the model using the WordDefinitionModel class. 
    If successful, it returns the trained Keras model; otherwise, it returns None."""

    wordDefinitionModel = WordDefinitionModel(
        batch_size=512,
        epochs=30
    )
    keras_model = wordDefinitionModel.kerasModel
    if keras_model is not None:
        print("Keras model initialized and loaded successfully.")
        return keras_model
    else:
        print("Keras model is not initialized. Running the training process...")
        await wordDefinitionModel.initializeAndTrainKerasModel()
        if wordDefinitionModel.kerasModel is not None:
            print("Keras model initialized and loaded successfully after training.")
            return wordDefinitionModel.kerasModel
        else:
            print("Failed to initialize and train the Keras model.")
            return None


async def main():
    kerasModel = await KerasModel()
    if kerasModel is not None:
        predicts = kerasModel.getPredictions("tasteless liquid that is essential for all known forms of life")
        print(f"Predicted word: {predicts[0]}, Predicted definition: {predicts[1]}")
    else:
        print("Keras model is not available. Cannot make predictions.")


asyncio.run(main())
