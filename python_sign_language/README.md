# Sign Language Detection Model Training

This project contains the training pipeline for a sign language detection model.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Place your sign language videos/images in a directory structure where each subdirectory is named after the sign it represents
   - Example structure:
     ```
     dataset/
     ├── hello/
     │   ├── video1.mp4
     │   └── video2.mp4
     ├── thank_you/
     │   ├── video1.mp4
     │   └── video2.mp4
     └── ...
     ```

4. Update the `data_dir` path in `train.py` to point to your dataset directory

5. Run the training:
```bash
python train.py
```

The trained model will be saved in the `models` directory as `sign_language_model.h5`.
Training history plots will be saved as `training_history.png`.