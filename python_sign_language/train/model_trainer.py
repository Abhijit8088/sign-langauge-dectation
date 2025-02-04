import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple, List
import numpy as np

class SignLanguageModelTrainer:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """Build and compile the CNN model"""
        model = models.Sequential([
            # CNN layers
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32) -> tf.keras.callbacks.History:
        """Train the model"""
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/sign_language_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
        )
        
    def save_model(self, path: str = 'models/sign_language_model.h5'):
        """Save the trained model"""
        self.model.save(path)