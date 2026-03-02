
import os
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

import pickle

from sentence_transformers import SentenceTransformer
from ai_edge_litert.interpreter import Interpreter
from tqdm import tqdm

# from src.utils import utilities as utils
from src.utils.SpacySplitter import SpacySplitter

transformer = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2')
#transformer = SentenceTransformer(model_name_or_path='all-mpnet-base-v2') # 768


class KerasModel:
    def __init__(self, model: tf.keras.Model, history: tf.keras.callbacks.History | None):
        self.model = model
        self.history = history 

    def getKerasModel(self) -> tf.keras.Model:
        return self.model

    def getHistory(self) -> tf.keras.callbacks.History | None:
        return self.history
    
    def getPredictions(self, input: str | list[str]) -> tuple[np.ndarray, np.ndarray]:
        embedding = transformer.encode(sentences=input, convert_to_numpy=True)
        definition_pred, word_pred = self.model.predict(embedding)
        predictions = np.argmax(definition_pred, axis=1), np.argmax(word_pred, axis=1)
        return predictions


class TFLiteModel:
    def __init__(self, model: Interpreter):
        self.model = model

    def getTFLiteModel(self) -> Interpreter | None:
        return self.model
    

class WordDefinitionModel:
    
    epochs: int
    batch_size: int

    inputs: np.ndarray
    definition_output_labels: np.ndarray
    word_output_labels: np.ndarray

    tfliteModel: TFLiteModel | None = None
    kerasModel: KerasModel | None = None
    
    spacy_splitter: SpacySplitter = SpacySplitter()


    def __init__(self, batch_size: int = 32, epochs: int = 10, dict = 'default'):
        r"""Initializes the WordDefinitionModel by loading the dataset, preparing the training data, and setting up paths for saving models and checkpoints. It also checks for available GPUs and clears any existing TensorFlow sessions to ensure a clean start.
        Args:
            batch_size (int): The number of samples per batch during training. Default is 32.
            epochs (int): The number of epochs to train the model. Default is 10.
            dict (str): The path to the CSV file containing words and their definitions or a named built-in dictionary. The CSV file should have 'word' and 'definition' columns. If no dictionary is provided, it will load a default dictionary from 'data/dictionaries/dict.csv'.
        """

        print("\n⏳ Initializing model and loading data...\n")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Available GPUs: {[gpu.name for gpu in gpus]}")        
        else:
            print("No GPUs found. Using CPU.")

        tf.keras.backend.clear_session()

        

        if dict == "default":
            print("⚠️  Using default dictionary. To use a custom dictionary, provide the path to a CSV file with 'word' and 'definition' columns when initializing the model.")
            df =pd.read_csv(os.getcwd() + '/data/datasets/default/default.csv')
        else:
            print(f"📂 Loading custom dictionary from {dict}...")
            df = pd.read_csv(dict)

        self.words = df['word'].tolist()
        self.definitions = df['definition'].tolist()
        self.word_labels = np.arange(len(self.definitions))
        self.definition_labels = np.arange(len(self.words))

        print(f"\n🔋 Loaded {len(self.words)} words with definitions.")

        self.epochs = epochs
        self.batch_size = batch_size
        self.ncols = 150

        self.checkpoints_path = os.getcwd() + "/data/checkpoints"
        self.keras_model_save_path = os.getcwd() + '/saved_models/word_definition_model.keras'
        self.keras_model_export_path = os.getcwd() + '/saved_models/SavedModel'
        self.tflite_model_save_path = os.getcwd() + '/saved_models/word_definition_model.tflite'

        self.tfliteModel = None
        self.kerasModel = None

        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(os.getcwd() + '/saved_models', exist_ok=True)

        if os.path.exists(self.keras_model_save_path):
            print("\n💽 Loading existing Keras model...\n")
            self.kerasModel = KerasModel(tf.keras.models.load_model(self.keras_model_save_path), None)



    async def save_checkpoint(self, data: np.ndarray | list, filename: str):
        r"""Saves data to a checkpoint file using pickle. This allows for resuming training or reusing preprocessed data without having to redo expensive computations.
        Args:
            data: The data to save (e.g., lists of augmented definitions, embeddings).
            filename (str): The name of the checkpoint file (e.g., 'augmented_definitions.pkl').
        """
        print(f"💾 Saving checkpoint: {filename}...")
        with open(f"{self.checkpoints_path}/{filename}", "wb") as f:
            pickle.dump(data, f)


    def load_checkpoint(self, filename: str) -> Any | None:
        r"""Loads data from a checkpoint file if it exists. This is useful for resuming training or reusing preprocessed data.
        Args:
            filename (str): The name of the checkpoint file (e.g., 'augmented_definitions.pkl').
        Returns:
            The data loaded from the checkpoint file, or None if the file does not exist.
        """

        path = f"{self.checkpoints_path}/{filename}"
        if os.path.exists(path):
            print(f"📂 Loading checkpoint: {filename}...")
            with open(path, "rb") as f:
                return pickle.load(f)
        return None
    

    async def save_and_backup_model(self, model: tf.keras.Model, history: tf.keras.callbacks.History):
        r"""Saves the trained Keras model and its training history to disk. This allows for later use without needing to retrain the model.
        Args:
            model (tf.keras.Model): The trained Keras model to save.
            history (tf.keras.callbacks.History): The training history to save for later analysis.
        """
        print(f"💾 Saving model to {self.keras_model_save_path}...")
        model.save(self.keras_model_save_path)
        self.kerasModel = KerasModel(model, history)

        print(f"💾 Exporting model to {self.keras_model_export_path} format...")
        model.export(self.keras_model_export_path)

        print(f"💾 Converting model to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_saved_model(self.keras_model_export_path)
        tflite_model = converter.convert()
        
        with open(self.tflite_model_save_path, "wb") as f:
            f.write(tflite_model)
        self.tfliteModel = TFLiteModel(Interpreter(model_path=self.tflite_model_save_path))

        print(f"💾 Saving Training History...")
        with open(f"{self.checkpoints_path}/training_history.pkl", "wb") as f:
            pickle.dump(history.history, f)


    def data_generator(self):
        r"""A generator function that yields batches of training data for the Keras model. It iterates through the combined dataset of word embeddings, definition embeddings, and augmented definition embeddings, yielding batches of inputs and their corresponding labels for both definition and word outputs.
        Yields:
            tuple: A tuple containing a batch of inputs and a dictionary of corresponding labels for definition and word outputs.
        """
        indices = np.arange(len(self.inputs))
        np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            x_batch = self.inputs[batch_indices]
            y_batch = {
                'definition_output': self.definition_output_labels[batch_indices],
                'word_output': self.word_output_labels[batch_indices]
            }
            yield x_batch, y_batch


    async def initializeAndTrainKerasModel(self) -> None:   #tuple[tf.keras.Model, tf.keras.callbacks.History | None] | None:
        r"""Initializes the Keras model by preparing the training data, defining the model architecture, compiling it, and starting the training process. It also handles saving the trained model and its history for later use."""
        try:
            # if os.path.exists(self.keras_model_export_path):
            #     print("\n💽 Loading existing model...\n")
            #     return (tf.keras.models.load_model(self.keras_model_save_path), None)
            if self.kerasModel is not None:
                print("\n💽 Keras model already initialized. Access it via the 'kerasModel' attribute.\n")
                # return (self.kerasModel.getKerasModel(), self.kerasModel.getHistory())
            else:
                print("\n🥣 Preparing Training Data...\n")
                
                AUGMENTED_DEFINITIONS = []
                AUGMENTED_DEFINITION_LABELS = []
                AUGMENTED_DEFINITION_EMBEDDINGS = []
                WORD_EMBEDDINGS = []
                DEFINITION_EMBEDDINGS = []
                
                try:
                    ad = self.load_checkpoint("augmented_definitions.pkl")
                    if ad is not None:
                        AUGMENTED_DEFINITIONS = ad
                    
                    adl = self.load_checkpoint("augmented_definition_labels.pkl")
                    if adl is not None:
                        AUGMENTED_DEFINITION_LABELS = adl

                    ade = self.load_checkpoint("augmented_definition_embeddings.pkl")
                    if ade is not None:
                        AUGMENTED_DEFINITION_EMBEDDINGS = ade

                    we = self.load_checkpoint("word_embeddings.pkl")
                    if we is not None:
                        WORD_EMBEDDINGS = we
                    
                    de = self.load_checkpoint("definition_embeddings.pkl")
                    if de is not None:
                        DEFINITION_EMBEDDINGS = de
                    
                except Exception as e:
                    print(f"❌ Error occurred while loading checkpoints: {e}")
                

                if len(AUGMENTED_DEFINITIONS) == 0:
                    pbar = tqdm(enumerate(self.definitions), total=len(self.definitions), ncols=self.ncols, desc="📚 Augmenting definitions")
                
                    for i, definition in pbar:
                        chunks = self.spacy_splitter.split_definition(definition)
                        for chunk in chunks:
                            AUGMENTED_DEFINITIONS.append(chunk)
                            AUGMENTED_DEFINITION_LABELS.append(i)

                    await self.save_checkpoint(AUGMENTED_DEFINITIONS, "augmented_definitions.pkl")
                    await self.save_checkpoint(AUGMENTED_DEFINITION_LABELS, "augmented_definition_labels.pkl")
                else:
                    print("\n🔄 Augmented definitions already exist. Skipping...")


                if len(AUGMENTED_DEFINITION_EMBEDDINGS) == 0:
                    augmented_defs = tqdm(range(0, len(AUGMENTED_DEFINITIONS), self.batch_size), ncols=self.ncols, desc="📝 Embedding Augments    ")

                    for i in augmented_defs:
                        batch = AUGMENTED_DEFINITIONS[i:i + self.batch_size]
                        embeddings = transformer.encode(batch, convert_to_numpy=True)
                        AUGMENTED_DEFINITION_EMBEDDINGS.append(embeddings)

                    AUGMENTED_DEFINITION_EMBEDDINGS = np.concatenate(AUGMENTED_DEFINITION_EMBEDDINGS, axis=0)
                    await self.save_checkpoint(AUGMENTED_DEFINITION_EMBEDDINGS, "augmented_definition_embeddings.pkl")
                else:
                    print("\n🔄 Augmented definition embeddings already exist. Skipping...")


                if len(WORD_EMBEDDINGS) == 0:
                    word_defs = tqdm(range(0, len(self.words), self.batch_size), ncols=self.ncols, desc="🔤 Embedding words       ")
                
                    for i in word_defs:
                        batch = self.words[i:i + self.batch_size]
                        embeddings = transformer.encode(batch, convert_to_numpy=True)
                        WORD_EMBEDDINGS.append(embeddings)

                    WORD_EMBEDDINGS = np.concatenate(WORD_EMBEDDINGS, axis=0)
                    await self.save_checkpoint(WORD_EMBEDDINGS, "word_embeddings.pkl")
                else:
                    print("\n🔄 Word embeddings already exist. Skipping...")


                if len(DEFINITION_EMBEDDINGS) == 0:
                    definition_defs = tqdm(range(0, len(self.definitions), self.batch_size), ncols=self.ncols, desc="📖 Embedding definitions ")
                
                    for i in definition_defs:
                        batch = self.definitions[i:i + self.batch_size]
                        embeddings = transformer.encode(batch, convert_to_numpy=True)
                        DEFINITION_EMBEDDINGS.append(embeddings)

                    DEFINITION_EMBEDDINGS = np.concatenate(DEFINITION_EMBEDDINGS, axis=0)
                    await self.save_checkpoint(DEFINITION_EMBEDDINGS, "definition_embeddings.pkl")
                else:
                    print("\n🔄 Definition embeddings already exist. Skipping...")



                temp_inputs = []
                temp_definition_output_labels = []
                temp_word_output_labels = []

                for i in range(len(WORD_EMBEDDINGS)):
                    temp_inputs.append(WORD_EMBEDDINGS[i])
                    temp_definition_output_labels.append(i)
                    temp_word_output_labels.append(i)

                for i in range(len(DEFINITION_EMBEDDINGS)):
                    temp_inputs.append(DEFINITION_EMBEDDINGS[i])
                    temp_definition_output_labels.append(i)
                    temp_word_output_labels.append(i)

                for i in range(len(AUGMENTED_DEFINITION_EMBEDDINGS)):
                    temp_inputs.append(AUGMENTED_DEFINITION_EMBEDDINGS[i])
                    temp_definition_output_labels.append(AUGMENTED_DEFINITION_LABELS[i])
                    temp_word_output_labels.append(AUGMENTED_DEFINITION_LABELS[i])

                self.inputs = np.array(temp_inputs)
                self.definition_output_labels = np.array(temp_definition_output_labels)
                self.word_output_labels = np.array(temp_word_output_labels)



                print(f"\n📊 Total samples: {len(self.inputs)} 📊\n")
                print(f"{self.inputs.shape[1]} features per sample. Preparing dataset for training...")
                print(f"\nEmbeddings::\n Words: {len(WORD_EMBEDDINGS)}" +
                      f" Definitions: {len(DEFINITION_EMBEDDINGS)}" + 
                      f" Augmented Definitions: {len(AUGMENTED_DEFINITION_EMBEDDINGS)})")
                


                input_layer = tf.keras.layers.Input(shape=(self.inputs.shape[1],), dtype=tf.float32)

                x = tf.keras.layers.Dense(512, activation='relu')(input_layer)
                # x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Dense(256, activation='relu')(x)
                # x = tf.keras.layers.BatchNormalization()(x)
                # x = tf.keras.layers.Dropout(0.3)(x)

                definition_output = tf.keras.layers.Dense(len(self.definitions), activation='softmax', name='definition_output')(x)
                word_output = tf.keras.layers.Dense(len(self.words), activation='softmax', name='word_output')(x)

                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=1e-3,
                    decay_steps=10000,
                    decay_rate=0.9
                )

                model = tf.keras.models.Model(inputs=input_layer, outputs=[definition_output, word_output])

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss={
                    'definition_output': 'sparse_categorical_crossentropy',
                    'word_output': 'sparse_categorical_crossentropy'
                    },
                    # loss_weights=[0.7, 0.3],
                    metrics={
                        'definition_output': 'accuracy', 
                        'word_output': 'accuracy'
                        }
                )

                dataset = tf.data.Dataset.from_generator(
                    self.data_generator,
                    output_signature=(
                        tf.TensorSpec(shape=(None, self.inputs.shape[1]), dtype=tf.float32),
                        {
                            'definition_output': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                            'word_output': tf.TensorSpec(shape=(None,), dtype=tf.int32)
                        }
                    )
                ).repeat().prefetch(tf.data.AUTOTUNE)

            

                # callbacks = [
                #     tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                #     tf.keras.callbacks.ModelCheckpoint(
                #         filepath=keras_model_save_path,
                #         save_best_only=True, # untill validation is setuup
                #         save_weights_only=False,
                #         mode='auto',
                #         verbose=1
                #     )
                # ]

                print("🧬 Begininning model fit...", flush=True)
                history = model.fit(
                    dataset,
                    # callbacks=callbacks
                    epochs=self.epochs,
                    steps_per_epoch = (len(self.inputs) + self.batch_size - 1) // self.batch_size,
                    validation_data=dataset,
                    validation_steps=(len(self.inputs) + self.batch_size - 1) // self.batch_size
                
                    # verbose=1
                )
        
                await self.save_and_backup_model(model, history)
                # return (model, history)
        except Exception as e:
                print(f"❌ Error during data preparation: {e}")
                return None
