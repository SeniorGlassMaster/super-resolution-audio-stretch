import math
import numpy as np
from librosa.core import resample

# Takes input mono audio track, stretches it to twice its length (interpolating
# samples through average of adjacent samples), and splits into arrays of
# window_size.
def preprocess_input(src, window_size=320):
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
        if not i % 100000 or i == double_length.size - 1:
            progress = 100 * ((i + 1) / double_length.size)
            if progress != 100:
                progress = str(progress)[:4]
                print('   Stretching audio: {}% complete   '.format(progress), end='\r')
            else:
                print('   Stretching audio: 100% complete   ')
    
    window_split = [double_length[i * window_size:(i + 1) * window_size] \
                    for i in range((double_length.size + window_size - 1) // window_size )]
    window_split = np.asarray(window_split)

    print('   Finished preprocess')
    return window_split

# Splits training output into arrays of window_size
def preprocess_output(src, window_size=320):
    window_split = [src[i * window_size:(i + 1) * window_size] \
                    for i in range((src.size + window_size - 1) // window_size )]
    window_split = np.asarray(window_split)

    return window_split

# def load_data():