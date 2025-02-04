import cv2
import mediapipe as mp
import numpy as np
from sign_detector import SignLanguageDetector
from sentence_processor import SentenceProcessor
from utils.visualization import draw_landmarks
from utils.gesture_db import load_gesture_database

def main():
    # Initialize detectors
    sign_detector = SignLanguageDetector()
    sentence_processor = SentenceProcessor()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Starting sign language detection... Press 'q' to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            continue
            
        # Mirror the frame for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Process frame and detect signs
        results = sign_detector.process_frame(frame)
        
        if results.hand_landmarks or results.face_landmarks:
            # Draw landmarks on frame
            frame = draw_landmarks(frame, results)
            
            # Detect sign and update sentence
            detected_sign = sign_detector.detect_sign(results)
            if detected_sign:
                sentence = sentence_processor.update_sentence(detected_sign)
                print(f"Current sentence: {sentence}")
            
        # Display frame
        cv2.imshow('Sign Language Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()