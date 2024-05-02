import csv

import numpy as np
import librosa.display
import json
import math
import librosa

from main import prepare_datasets, build_model
from params import checkpoint_path, JSON_TEST_PATH, DATASET_TEST_PATH, SAMPLE_RATE, SAMPLES_PER_TRACK, num_mfcc, n_fft, hop_length, num_segments

def save_mfcc(dataset_path, json_path, num_mfcc, n_fft, hop_length, num_segments):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mfcc": [],
        "spectrogram": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i in range(250):

        #print(dataset_path + "/f_"+str(i)+".wav")
        signal, sample_rate = librosa.load(dataset_path + "/f_"+str(i)+".wav", sr=SAMPLE_RATE)

        # process all segments of audio file
        for d in range(num_segments):

            # Calculate start and finish sample for current segment
            start = samples_per_segment * d
            finish = start + samples_per_segment
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T
            # Extract spectrogram with smaller resolution
            spectrogram = librosa.amplitude_to_db(librosa.stft(signal[start:finish], n_fft=n_fft//64, hop_length=hop_length), ref=np.max)
            spectrogram = spectrogram.T
            # Store only MFCCs and spectrograms with expected number of vectors
            if len(mfcc) == num_mfcc_vectors_per_segment:
                data["mfcc"].append(mfcc.tolist())
                data["spectrogram"].append(spectrogram.tolist())

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X_mfcc = np.array(data["mfcc"])
    X_spectrogram = np.array(data["spectrogram"])

    X = np.concatenate((X_mfcc, X_spectrogram), axis=2)

    return X

if __name__ == "__main__":

    # get train, validation, test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # load the model
    model.load_weights(checkpoint_path)

    # generate data
    save_mfcc(DATASET_TEST_PATH, JSON_TEST_PATH, num_mfcc, n_fft, hop_length, num_segments)

    # load data
    X = load_data(JSON_TEST_PATH)

    # predict samples
    X = X[..., np.newaxis] # array shape (1, 130, 13, 1)

    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)

    with open('./resources/submission.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'TARGET'])
        for i, pred in enumerate(predicted_index):  # Default start index is 0
            writer.writerow([i, pred])