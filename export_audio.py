from librosa.output import write_wav
from parameters import *

def render_audio(model_output, path, sample_rate, window_size=WINDOW_SIZE, overlap=OVERLAP):
    rendered = np.zeros(window_size * model_output.shape[0])
    r_index = 0
    for i in range(model_output.shape[0]):
        for j in range(model_output[i].shape[0]):
            if j < overlap:
                if i > 0:
                    rendered[r_index + j] = rendered[r_index + j] + \
                                            (model_output[i][j] * (j / float(overlap)))
            elif j >= (window_size - overlap - 1):
                if i < model_output.shape[0]:
                    rendered[r_index + j] = rendered[r_index + j] + \
                                            (model_output[i][j] * ((window_size - j - 1) / float(overlap)))
            else:
                rendered[r_index + j] = model_output[i][j]
        r_index += (window_size - overlap - 1)
    rendered = np.array(rendered)
    write_wav(path, rendered, sample_rate)