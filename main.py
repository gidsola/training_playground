
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

import asyncio

from src.models.WordDefinitionModel import WordDefinitionModel


async def main():
    wordDefinitionModel = WordDefinitionModel()
    
    if wordDefinitionModel.kerasModelHandler is not None:
        kerasHandler = wordDefinitionModel.kerasModelHandler
        kerasModel = kerasHandler.getKerasModel() # model direct
    
        result = kerasHandler.getPredictions("tasteless liquid that is essential for all known forms of life")
        if result is not None:
            print("\nPredictions:")
            print(f"\nWord: {result['Word']}, \nDefinition: {result['Definition']}")

        print("\nDisplaying weight information:\n")
        for weight in kerasModel.weights:
            print(f"Weight name: {weight.name}, Weight shape: {weight.shape}")
        print()
    else:
        print("Keras model not found... Training model for use...")
        await wordDefinitionModel.createKerasModel()


asyncio.run(main())
