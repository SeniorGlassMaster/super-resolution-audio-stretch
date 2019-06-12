from librosa.output import write_wav
from parameters import *
import numpy as np

def render_audio(model_output, path, sample_rate, window_size=WINDOW_SIZE, overlap=OVERLAP):
    rendered = np.zeros(window_size * model_output.shape[0])
    r_index = 0
    for i in range(model_output.shape[0]):
        for j in range(model_output[i].shape[0] - overlap):
            if j < overlap:
                if i > 0:
                    amp_factor = j / float(overlap)
                    rendered[r_index + j] = (model_output[i][j] * amp_factor) + (model_output[i-1][model_output[i-1].shape[0] - 1 - overlap + j] * (1-amp_factor))
            else:
                rendered[r_index + j] = float(model_output[i][j])
        r_index += (window_size - overlap - 1)
    rendered = np.array(rendered)
    write_wav(path, rendered, sample_rate)