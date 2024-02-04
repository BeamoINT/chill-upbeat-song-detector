import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import json

song_paths = [
    "/songs/song1.mp3", "/songs/song2.mp3", "/songs/song3.mp3", "/songs/song4.mp3", 
    "/songs/song5.mp3", "/songs/song6.mp3", "/songs/song7.mp3", "/songs/song8.mp3",
    "/songs/song9.mp3", "/songs/song10.mp3", "/songs/song11.mp3", "/songs/song12.mp3",
    "/songs/song13.mp3", "/songs/song14.mp3", "/songs/song15.mp3", "/songs/song16.mp3",
    "/songs/song17.mp3", "/songs/song18.mp3", "/songs/song19.mp3", "/songs/song20.mp3",
    "/songs/song21.mp3", "/songs/song22.mp3", "/songs/song23.mp3", "/songs/song24.mp3",
    "/songs/song25.mp3", "/songs/song26.mp3", "/songs/song27.mp3", "/songs/song28.mp3",
    "/songs/song29.mp3", "/songs/song30.mp3", "/songs/song31.mp3", "/songs/song32.mp3",
    "/songs/song33.mp3", "/songs/song34.mp3", "/songs/song35.mp3", "/songs/song36.mp3",
    "/songs/song37.mp3", "/songs/song38.mp3", "/songs/song39.mp3", "/songs/song40.mp3",
    "/songs/song41.mp3", "/songs/song42.mp3", "/songs/song43.mp3", "/songs/song44.mp3",
    "/songs/song45.mp3", "/songs/song46.mp3", "/songs/song47.mp3", "/songs/song48.mp3",
    "/songs/song49.mp3", "/songs/song50.mp3", "/songs/song51.mp3", "/songs/song52.mp3",
    "/songs/song53.mp3", "/songs/song54.mp3", "/songs/song55.mp3", "/songs/song56.mp3",
    "/songs/song57.mp3", "/songs/song58.mp3", "/songs/song59.mp3", "/songs/song60.mp3"
]

# Function to load ratings from a JSON file
def load_ratings_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        ratings = sum(data['song_ratings'], [])
    return ratings

# Load the ratings
chill_to_crazy_ratings = load_ratings_from_json('/songratings.json')

# Function to extract MFCC features from an MP3 song
def extract_features(song_path):
    try:
        audio, sample_rate = librosa.load(song_path, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", song_path, "\nError: ", e)
        return None 
    return mfccs_processed

# Extract features from each song
features = []
for path in song_paths:
    mfccs = extract_features(path)
    if mfccs is not None:
        features.append(mfccs)

# Ensure the features and labels have the same length
if len(features) != len(chill_to_crazy_ratings):
    raise ValueError("The number of features and labels must be the same.")

# Convert to numpy arrays
X = np.array(features)
y = np.array(chill_to_crazy_ratings)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and train the model
model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
model.fit(X_train, y_train)

# Predict and evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")