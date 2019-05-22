import math
import numpy as np
from librosa.core import resample
from hyperparameters import *

# Takes input mono audio track, stretches it to twice its length (interpolating
# samples through average of adjacent samples), and splits into arrays of
# window_size.
def preprocess_input_data(src, window_size=WINDOW_SIZE, overlap=OVERLAP):
    if not isinstance(src.size, int):
        raise ValueError("Non-mono track input.")
    
    print("Preprocessing input...")
    double_length = np.zeros(2 * len(src))
    for i in range(double_length.size):   
        # Interpolate audio 
        if i % 2:
            if math.ceil(i / 2.0) < src.size:
                double_length[i] = (src[int(math.floor(i / 2.0))] \
                                    + src[int(math.ceil(i / 2.0))]) / 2.0
        else:
            double_length[i] = src[int(i / 2)]
        
        # Print progress
        if i % 100000 == 0 or i == double_length.size - 1:
            progress = 100 * ((i + 1) / double_length.size)
            if progress != 100:
                progress = str(progress)[:4]
                print('   Stretching audio: {}% complete   '.format(progress), end='\r')
            else:
                print('   Stretching audio: 100% complete   ')
    
    # window_split = [double_length[i * window_size:(i + 1) * window_size] \
    #                 for i in range((double_length.size + window_size - 1) // window_size )]
    # window_split = np.asarray(window_split)

    window_split = window_splitter(double_length, window_size, overlap)
    print('   Finished preprocess')
    return window_split

# Splits target audio into arrays of window_size
def preprocess_target_data(src, window_size=WINDOW_SIZE, overlap=OVERLAP):
    return window_splitter(src, window_size, overlap)

def window_splitter(src, window_size=WINDOW_SIZE, overlap=OVERLAP):
    window_split = []
    i = 0
    while i < src.size:
        window_split.append(src[i:(i+window_size)])
        i += window_size - overlap - 1
    return np.array(window_split)