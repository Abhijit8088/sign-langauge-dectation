import os
import cv2
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split

class SignLanguageDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.classes = self._get_classes()
        
    def _get_classes(self) -> List[str]:
        """Get all class names from the data directory"""
        return sorted([d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))])
        
    def load_data(self, img_size: Tuple[int, int] = (224, 224)) -> Tuple:
        """Load and preprocess all images and labels"""
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        img = img / 255.0  # Normalize
                        images.append(img)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    
        return np.array(images), np.array(labels)
        
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """Split data into training and testing sets"""
        X, y = self.load_data()
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, stratify=y)