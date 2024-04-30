import librosa.display
import json
import os
import math
import librosa
import numpy as np

from params import DATASET_TRAIN_PATH, JSON_TRAIN_PATH, SAMPLE_RATE, SAMPLES_PER_TRACK, num_mfcc, n_fft, hop_length, num_segments

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
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # normalize signal
                signal = librosa.util.normalize(signal)

                # force 5 seconds of audio
                desired_length = 5 * sample_rate  # 5 seconds
                if len(signal) < desired_length:
                    signal = np.pad(signal, (0, desired_length - len(signal)), mode='constant')
                elif len(signal) > desired_length:
                    signal = signal[:desired_length]

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        lbl = (file_path.split('-')[-1]).split('.')[0]
                        data["labels"].append(int(lbl))
                        print("{}, segment:{}, label:{}".format(file_path, d+1,lbl))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_TRAIN_PATH, JSON_TRAIN_PATH, num_mfcc, n_fft, hop_length, num_segments)