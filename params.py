ROOT_PATH = "./resources"
DATASET_TRAIN_PATH = ROOT_PATH + "/ESC_train"
DATASET_TEST_PATH = ROOT_PATH + "/ESC_test"
JSON_TRAIN_PATH = ROOT_PATH + "/data_train.json"
JSON_TEST_PATH = ROOT_PATH + "/data_test.json"
checkpoint_path = ROOT_PATH + "/cp.weights.h5"
SAMPLE_RATE = 10000
num_segments = 1
num_mfcc = 128
n_fft = 2048
hop_length = 1024
TRACK_DURATION = 5 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION