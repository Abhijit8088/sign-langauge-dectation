import json
from typing import Dict
import os

def load_gesture_database() -> Dict:
    """
    Loads the gesture database from a JSON file.
    Creates a default database if none exists.
    """
    db_path = "data/gesture_db.json"
    
    if not os.path.exists(db_path):
        default_db = {
            "HELLO": {
                "hands": 1,
                "description": "Wave hand near face",
                "required_landmarks": [0, 4, 8, 12, 16, 20]
            },
            "THANK_YOU": {
                "hands": 1,
                "description": "Flat hand moving down from chin",
                "required_landmarks": [0, 4, 8, 12, 16, 20]
            },
            "LOVE": {
                "hands": 2,
                "description": "Cross arms over chest",
                "required_landmarks": [0, 4, 8, 12, 16, 20]
            },
            # Add more gestures as needed
        }
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with open(db_path, 'w') as f:
            json.dump(default_db, f, indent=2)
            
        return default_db
        
    with open(db_path, 'r') as f:
        return json.load(f)