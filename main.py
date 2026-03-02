
import asyncio

import numpy as np
import pandas as pd
import tensorflow as tf

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

    # if keras_model is not None:
    #     print("✅ Keras model is ready for predictions.")
    #     keras_model.model.summary()


    #     keraspredictions = keras_model.getPredictions(input)
    #     print(f"Keras Predictions: {keraspredictions}")
        # return keraspredictions

predicts = asyncio.run(prediction_Keras("liquid covering most of the earths surface"))
print(f"Predictions: {predicts}")


# def prediction_Keras(input: str) -> str | tuple[str, tf.keras.callbacks.History | None]:
#     r"""Generates a prediction using the Keras model for a given input string. The function determines whether to predict a word or a definition based on the input length.
#     Args:
#         input (str): The input string for which to generate a prediction.
#     Returns:
#         str: The predicted word or definition.
#     """
#     result = getKerasModel()
#     if result is None:
#         return "❌ Keras model not available."
    
#     keras_model, history = result

    
    
#     embedding = transformer.encode([input], convert_to_numpy=True)
#     definition_pred, word_pred = keras_model.predict(embedding)
    
#     if(len(str.split(input))) > 1:
#         predicted_idx = np.argmax(word_pred, axis=1)[0]
#         return words[predicted_idx], history
    
#     predicted_idx = np.argmax(definition_pred, axis=1)[0]
#     return definitions[predicted_idx], history


# def prediction_TFLite(input: str) -> str:
#     r"""Generates a prediction using the TFLite model for a given input string. The function determines whether to predict a word or a definition based on the input length.
#     Args:
#         input (str): The input string for which to generate a prediction.
#     Returns:
#         str: The predicted word or definition.
#     """
#     interpreter = getTFLiteModel()
#     if interpreter is None:
#         return "❌ TFLite model not available."
    
#     embedding = transformer.encode([input], convert_to_numpy=True).astype(np.float32)
#     interpreter.allocate_tensors()
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     interpreter.set_tensor(input_details[0]['index'], embedding)
#     interpreter.invoke()

#     if len(input.split()) > 1:
#         word_output = interpreter.get_tensor(output_details[1]['index'])
#         predicted_idx = np.argmax(word_output)
#         return words[predicted_idx]
#     else:
#         definition_output = interpreter.get_tensor(output_details[0]['index'])
#         predicted_idx = np.argmax(definition_output)
#         return definitions[predicted_idx]



# prediction, history = prediction_Keras("liquid covering most of the earths surface")
# print(f"Word from description: {prediction}")
# plot_training_history(history)
# print(prediction_TFLite("It covers about 71% of the Earth's surface"))
# print(prediction_TFLite("wet"))
