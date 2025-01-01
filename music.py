import streamlit as st
import librosa
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load the trained SVM model
with open("musics.pkl", "rb") as file:
    model = pickle.load(file)

# Feature extraction function
def extract_features(audio_path):
    features = {}
    
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=3)

    # Extract different audio features
    features['chroma_stft_mean'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    features['chroma_stft_var'] = np.var(librosa.feature.chroma_stft(y=y, sr=sr))
    features['rms_mean'] = np.mean(librosa.feature.rms(y=y))
    features['rms_var'] = np.var(librosa.feature.rms(y=y))
    features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_centroid_var'] = np.var(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['spectral_bandwidth_var'] = np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['rolloff_var'] = np.var(librosa.feature.spectral_rolloff(y=y, sr=sr))
    features['zero_crossing_rate_mean'] = np.mean(librosa.feature.zero_crossing_rate(y=y))
    features['zero_crossing_rate_var'] = np.var(librosa.feature.zero_crossing_rate(y=y))
    features['harmony_mean'] = np.mean(librosa.effects.harmonic(y))
    features['harmony_var'] = np.var(librosa.effects.harmonic(y))
    features['perceptr_mean'] = np.mean(librosa.effects.percussive(y))
    features['perceptr_var'] = np.var(librosa.effects.percussive(y))
    
    # Get tempo using onset envelope to avoid errors
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features['tempo'] = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = np.mean(mfccs[i-1])
        features[f'mfcc{i}_var'] = np.var(mfccs[i-1])
    
    return pd.DataFrame([features])

# Feature standardization function
def standardize_features(features_df):
    # Ensure the features_df is a DataFrame
    if isinstance(features_df, np.ndarray):
        features_df = pd.DataFrame(features_df)
    
    # Define the expected columns
    columns = [
        'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 
        'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var', 'zero_crossing_rate_mean', 
        'zero_crossing_rate_var', 'harmony_mean', 'harmony_var', 'perceptr_mean', 
        'perceptr_var', 'tempo'
    ] + [f'mfcc{i}_mean' for i in range(1, 21)] + [f'mfcc{i}_var' for i in range(1, 21)]
    
    # Ensure all columns exist in the DataFrame
    missing_columns = [col for col in columns if col not in features_df.columns]
    if missing_columns:
        print(f"Warning: Missing columns: {missing_columns}")
    
    # Select only the available columns
    available_columns = [col for col in columns if col in features_df.columns]
    if not available_columns:
        raise ValueError("No valid columns found for feature standardization!")

    features_df = features_df[available_columns]
    
    # Initialize the scaler and fit it to the extracted features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features_df)
    
    return standardized_features

# Streamlit app
st.title("Music Genre Classification")
st.write("Upload a `.wav` audio file to predict its genre and visualize results.")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner("Processing the audio file..."):
        # Save uploaded file temporarily
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        
        # Extract features
        extracted_features = extract_features("temp.wav")
        st.write("Extracted Features:", extracted_features)
        
        # Standardize features
        standardized_features = standardize_features(extracted_features)
        
        # Predict genre
        prediction = model.predict(standardized_features)
        probabilities = model.decision_function(standardized_features)

        # Visualization
        genres = ["blues", "classical", "country", "disco", "jazz", 
                  "pop", "hiphop", "metal", "reggae", "rock"]
        probabilities = probabilities[0]
        genre_probabilities = {genres[i]: probabilities[i] for i in range(len(genres))}
        
        st.write(f"Predicted Genre: **{prediction[0]}**")
        
        # Bar plot for genre probabilities with different colors
        plt.figure(figsize=(10, 6))
        color_map = sns.color_palette("Set2", len(genres))
        plt.bar(genre_probabilities.keys(), genre_probabilities.values(), color=color_map)
        plt.xlabel("Genres")
        plt.ylabel("Confidence Score")
        plt.title("Genre Prediction Confidence")
        st.pyplot(plt)

        # Creative Radar Chart for Genre Prediction
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        labels = list(genre_probabilities.keys())
        values = list(genre_probabilities.values())
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]  # To close the loop in the radar chart
        angles += angles[:1]  # To close the loop in the radar chart
        
        ax.fill(angles, values, color='skyblue', alpha=0.25)
        ax.plot(angles, values, color='skyblue', linewidth=2)
        
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title("Genre Prediction Radar Chart", size=15)
        
        st.pyplot(fig)

        # Heatmap of genre prediction confidence
        heatmap_data = np.array(list(genre_probabilities.values())).reshape(1, -1)
        heatmap_labels = list(genre_probabilities.keys())
        
        fig, ax = plt.subplots(figsize=(10, 2))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=heatmap_labels, yticklabels=["Confidence"], cbar_kws={'label': 'Confidence Score'})
        ax.set_title("Genre Confidence Heatmap", size=15)
        st.pyplot(fig)

    st.success("Prediction Complete!")
