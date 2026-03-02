
import matplotlib.pyplot as plt

def plot_training_history(history, filename='training_curves.png'):
    r"""Plots the training and validation loss and accuracy curves for both outputs.
    Expects the history object to contain the following keys:
    - 'definition_output_loss'
    - 'val_definition_output_loss'
    - 'word_output_loss'
    - 'val_word_output_loss'
    - 'definition_output_accuracy'
    - 'val_definition_output_accuracy'
    - 'word_output_accuracy'
    - 'val_word_output_accuracy'
    
    Args:
        history: The history object returned by model.fit() containing training metrics.
        filename: The name of the file to save the plot to.
    """

    if history is None:
        print("No training history available to plot.")
        return
    try:
        required_keys = [
            'definition_output_loss', 'val_definition_output_loss',
            'word_output_loss', 'val_word_output_loss',
            'definition_output_accuracy', 'val_definition_output_accuracy',
            'word_output_accuracy', 'val_word_output_accuracy'
        ]
        for key in required_keys:
            if key not in history.history:
                print(f"Warning: '{key}' not found in history. Skipping plot for this metric.")
                return
            
    except Exception as e:
        print(f"Error accessing history metrics: {e}")
        return
    
    plt.figure(figsize=(12, 8))
    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['definition_output_loss'], label='Train Definition Loss')
    plt.plot(history.history['val_definition_output_loss'], label='Val Definition Loss')
    plt.title('Definition Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history['word_output_loss'], label='Train Word Loss')
    plt.plot(history.history['val_word_output_loss'], label='Val Word Loss')
    plt.title('Word Loss')
    plt.legend()

    # Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history.history['definition_output_accuracy'], label='Train Definition Accuracy')
    plt.plot(history.history['val_definition_output_accuracy'], label='Val Definition Accuracy')
    plt.title('Definition Accuracy')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(history.history['word_output_accuracy'], label='Train Word Accuracy')
    plt.plot(history.history['val_word_output_accuracy'], label='Val Word Accuracy')
    plt.title('Word Accuracy')
    plt.legend()

    try:
        plt.suptitle('Training and Validation Curves')
        plt.tight_layout()
    except Exception as e:
        print(f"Error setting plot title or layout: {e}")
   
        plt.savefig(filename)
        plt.show()
  
