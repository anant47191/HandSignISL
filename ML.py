import argparse
import numpy as np
import pandas as pd
import os
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class MudraRecognizer:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.mp_hands = mp.solutions.hands
        # initialize Hands with recommended parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def load_data(self, dataset_root="dataset"):
        """Load CSV data saved by training.py under dataset/<label>/<label>.csv

        Supports both the older flatten CSVs (files directly in a folder) and
        the newer per-label layout created by training.py.
        Returns X (ndarray) and y (np.ndarray of labels)
        """
        data = []
        labels = []

        if not os.path.exists(dataset_root):
            print(f"Error: dataset root '{dataset_root}' not found!")
            return None, None

        # Two patterns: dataset/<label>/<label>.csv OR dataset/<label>.csv
        # iterate directory entries
        for entry in os.listdir(dataset_root):
            entry_path = os.path.join(dataset_root, entry)
            # If folder, look for a csv named <entry>.csv inside
            if os.path.isdir(entry_path):
                csv_file = os.path.join(entry_path, f"{entry}.csv")
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file, header=0)
                    # If header exists, drop 'class' column if present otherwise drop last
                    if 'class' in df.columns:
                        X = df.drop(columns=['class']).values
                    else:
                        X = df.iloc[:, :-1].values

                    data.append(X)
                    labels.extend([entry] * len(X))
                    continue

            # otherwise, if it's a CSV file directly inside dataset_root
            if entry.lower().endswith('.csv'):
                label_name = os.path.splitext(entry)[0]
                csv_file = entry_path
                df = pd.read_csv(csv_file, header=0)
                if 'class' in df.columns:
                    X = df.drop(columns=['class']).values
                else:
                    X = df.iloc[:, :-1].values

                data.append(X)
                labels.extend([label_name] * len(X))

        if not data:
            print("No data found in dataset/ — collect samples first (training.py) or place CSVs manually.")
            return None, None

        # ensure all arrays have same number of columns -> pad shorter arrays with zeros
        max_cols = max(a.shape[1] for a in data)
        padded = []
        for a in data:
            if a.shape[1] < max_cols:
                pad_width = max_cols - a.shape[1]
                a = np.hstack([a, np.zeros((a.shape[0], pad_width), dtype=a.dtype)])
            elif a.shape[1] > max_cols:
                a = a[:, :max_cols]
            padded.append(a)

        X = np.vstack(padded)
        y = np.array(labels)
        return X, y
    
    def preprocess_data(self, X, y):
        """Preprocess the data and encode labels"""
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X, y_encoded

    
    def train(self, test_size=0.2, random_state=42):
        """Train the mudra recognition model"""
        print("Loading data (looking in 'dataset/' then 'data/')...")
        X, y = self.load_data('dataset')
        if X is None:
            X, y = self.load_data('data')
        
        if X is None:
            return False
            
        print(f"Loaded {len(X)} samples with {len(set(y))} classes")
        print(f"Classes: {set(y)}")
        
        print("Preprocessing data...")
        X, y_encoded = self.preprocess_data(X, y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        self.model.fit(X_train, y_train)
        # save feature count for prediction checks
        try:
            self.n_features_in_ = self.model.n_features_in_
        except Exception:
            self.n_features_in_ = X_train.shape[1]
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return True

    def get_dataset_labels(self, dataset_root="dataset"):
        """Return sorted list of dataset labels (folder names with csv)"""
        labels = []
        if not os.path.exists(dataset_root):
            return labels
        for entry in os.listdir(dataset_root):
            entry_path = os.path.join(dataset_root, entry)
            if os.path.isdir(entry_path):
                csv_file = os.path.join(entry_path, f"{entry}.csv")
                if os.path.exists(csv_file):
                    labels.append(entry)
            elif entry.lower().endswith('.csv'):
                labels.append(os.path.splitext(entry)[0])

        return sorted(labels)
    
    def save_model(self, filepath="mudra_model.pkl"):
        """Save the trained model to a file"""
        if self.model is None:
            print("No model to save. Train the model first.")
            return False
            
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder
            }, f)
            
        print(f"Model saved to {filepath}")
        return True
    
    def load_model(self, filepath="mudra_model.pkl"):
        """Load a trained model from a file"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found!")
            return False
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        try:
            self.n_features_in_ = int(getattr(self.model, 'n_features_in_', None) or 0)
        except Exception:
            self.n_features_in_ = None
        
        print(f"Model loaded from {filepath}")
        return True
    
    def predict(self, landmarks):
        """Predict the mudra from hand landmarks"""
        if self.model is None:
            print("No model loaded. Train or load a model first.")
            return None
            
        # Ensure landmarks are in the right format. Accept either single-hand (63) or two-hand (126).
        if not hasattr(self, 'n_features_in_'):
            # If model not trained or loaded, infer nothing
            expected = None
        else:
            expected = int(self.n_features_in_)

        L = len(landmarks)
        # If model expects a different shape, try to adapt by padding or truncating
        if expected is not None and L != expected:
            if L < expected:
                # pad with zeros
                landmarks = list(landmarks) + [0.0] * (expected - L)
                print(f"Warning: input landmarks length {L} mismatched model features {expected}, padding with zeros.")
            else:
                # truncate
                landmarks = list(landmarks)[:expected]
                print(f"Warning: input landmarks length {L} mismatched model features {expected}, truncating.")
            
        # Make prediction
        prediction = self.model.predict([landmarks])
        label = self.label_encoder.inverse_transform(prediction)
        confidence = 0.0
        try:
            probs = self.model.predict_proba([landmarks])[0]
            confidence = float(max(probs))
        except Exception:
            confidence = 0.0

        return label[0], confidence
    
    def real_time_detection(self):
        """Run real-time mudra detection using webcam"""
        if self.model is None:
            print("No model loaded. Train or load a model first.")
            return
            
        cap = cv2.VideoCapture(0)
        
        while True:
            success, img = cap.read()
            if not success:
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            prediction_text = "No hand detected"

            # Build a two-hand landmark vector consistent with training: LEFT then RIGHT (each 21*3)
            if results.multi_hand_landmarks:
                # initialize 126-length vector with zeros
                vector = [0.0] * (21 * 3 * 2)

                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = None
                    if results.multi_handedness and len(results.multi_handedness) > i:
                        try:
                            label = results.multi_handedness[i].classification[0].label
                        except Exception:
                            label = None

                    base = 0
                    if label and label.lower() == 'right':
                        base = 63

                    for j, lm in enumerate(hand_landmarks.landmark):
                        vector[base + j*3] = lm.x
                        vector[base + j*3 + 1] = lm.y
                        vector[base + j*3 + 2] = lm.z

                    # Draw for feedback
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Make prediction with the full two-hand vector
                try:
                    mudra, conf = self.predict(vector)
                    if mudra:
                        prediction_text = f"{mudra} ({conf:.2f})"
                except Exception as e:
                    prediction_text = f"Error: {e}"
            
            # Display prediction
            cv2.putText(img, prediction_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Mudra Recognition", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run mudra recognition")
    parser.add_argument("--train", action="store_true", help="Train model from dataset/ and save model")
    parser.add_argument("--detect", action="store_true", help="Run real-time detection using webcam")
    parser.add_argument("--model", type=str, default="mudra_model.pkl", help="Path to load/save model")
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset root folder")
    parser.add_argument("--retrain-if-new", action="store_true", help="If a model exists but dataset contains new labels, retrain automatically")
    args = parser.parse_args()

    recognizer = MudraRecognizer()
    model_path = args.model

    # If training was requested, run training and save
    if args.train:
        print("Training model from dataset...")
        if recognizer.train():
            recognizer.save_model(model_path)
        else:
            print("Training failed: make sure dataset/ has CSV files (see training.py)")

    # If model exists, load it
    model_loaded = False
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        model_loaded = recognizer.load_model(model_path)

    # Check for new labels in dataset; if model exists and new labels appear -> retrain
    dataset_labels = recognizer.get_dataset_labels(args.dataset)
    if model_loaded and dataset_labels and recognizer.label_encoder is not None:
        model_labels = list(recognizer.label_encoder.classes_)
        new_labels = set(dataset_labels) - set(model_labels)
        if new_labels:
            print(f"Dataset contains new labels that are not in the loaded model: {sorted(new_labels)}")
            if args.retrain_if_new:
                print("Retraining because --retrain-if-new was specified.")
                if recognizer.train():
                    recognizer.save_model(model_path)
                    model_loaded = True
            else:
                # interactive choice
                choice = input("Retrain model to include new labels? (y/N): ")
                if choice.strip().lower().startswith('y'):
                    if recognizer.train():
                        recognizer.save_model(model_path)
                        model_loaded = True

    # If model not loaded and not training requested: try to train automatically
    if not model_loaded and not args.train:
        print("No usable model loaded — training from dataset now.")
        if recognizer.train():
            recognizer.save_model(model_path)
            model_loaded = True

    # If detect flag specified, run detection
    if args.detect:
        if not model_loaded:
            print("No model available for detection. Train first (--train) or allow automatic training.")
        else:
            recognizer.real_time_detection()

    # Default behaviour: if no flags provided, run detection (interactive)
    if not any([args.train, args.detect]):
        if model_loaded:
            print("Starting real-time detection (press q to quit)")
            recognizer.real_time_detection()
        else:
            print("No model available and no --train flag provided. Use --train to build a model from dataset/ or add training data with training.py")