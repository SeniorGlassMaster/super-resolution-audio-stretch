import math
import numpy as np
from parameters import WINDOW_SIZE, OVERLAP

# Takes input mono audio track, stretches it to twice its length (interpolating
# samples through average of adjacent samples), and splits into arrays of
# window_size.
def preprocess_input_data(src, window_size=WINDOW_SIZE, overlap=OVERLAP):
    if not isinstance(src.shape[0], int):
        raise ValueError("Non-mono track input.")
    
    print("Preprocessing input...")
    double_length = np.zeros(2 * len(src))
    for i in range(double_length.shape[0]):   
        # Interpolate audio 
        if i % 2:
            if math.ceil(i / 2.0) < src.shape[0]:
                double_length[i] = (src[int(math.floor(i / 2.0))] \
                                    + src[int(math.ceil(i / 2.0))]) / 2.0
        else:
            double_length[i] = src[int(i / 2)]
        
        # Print progress
        if i % 100000 == 0 or i == double_length.shape[0] - 1:
            progress = 100 * ((i + 1) / double_length.shape[0])
            if progress != 100:
                progress = str(progress)[:4]
                print('   Stretching audio: {}% complete   '.format(progress), end='\r')
            else:
                print('   Stretching audio: 100% complete   ')

    window_split = window_splitter(double_length, window_size, overlap)
    print('   Finished preprocess')
    return window_split

# Splits target audio into arrays of window_size
def preprocess_target_data(src, window_size=WINDOW_SIZE, overlap=OVERLAP):
    return window_splitter(src, window_size, overlap)

def window_splitter(src, window_size=WINDOW_SIZE, overlap=OVERLAP):
    window_split = []
    i = 0
    while i < (src.shape[0] - window_size):
        window_split.append(src[i:(i+window_size)])
        i += window_size - overlap - 1
    return np.array(window_split)