import os
from data_loader import SignLanguageDataLoader
from model_trainer import SignLanguageModelTrainer
import matplotlib.pyplot as plt

def plot_training_history(history):
    """Plot training & validation accuracy and loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])
    
    # Loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize data loader
    data_dir = "path/to/your/dataset"  # Replace with your dataset path
    loader = SignLanguageDataLoader(data_dir)
    
    # Load and split data
    X_train, X_test, y_train, y_test = loader.split_data()
    
    # Initialize and train model
    trainer = SignLanguageModelTrainer(num_classes=len(loader.classes))
    history = trainer.train(X_train, y_train, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    trainer.save_model()
    
    # Print final metrics
    test_loss, test_acc = trainer.model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()