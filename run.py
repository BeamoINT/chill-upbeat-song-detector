import librosa
import numpy as np
import tensorflow as tf
from joblib import load
import os

ai_model_dir = 'AI-Model'

model_path = os.path.join(ai_model_dir, 'modelweights.h5')
model = tf.keras.models.load_model(model_path)

def extract_features(song_path):
    audio, sample_rate = librosa.load(song_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

scaler_path = os.path.join(ai_model_dir, 'scaler.save')
scaler = load(scaler_path)

song_path = 'test.mp3'

features = extract_features(song_path).reshape(1, -1)

features = scaler.transform(features)

predicted_rating = model.predict(features)

predicted_rating = np.clip(np.round(predicted_rating), 1, 5)
print(f"Predicted 'chill to crazy' rating for the song is: {predicted_rating[0][0]}")