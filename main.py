import math
import numpy as np
from librosa.core import load
from librosa.output import write_wav
import load_data

def main():
    input_audio, sr = load("./midi_renders/fugue_1_plucks.wav")
    output_audio, sr = load("./midi_renders/fugue_1_plucks_slow.wav")
    input_audio = load_data.preprocess_input(input_audio)
    output_audio = load_data.preprocess_output(output_audio)
    assert(input_audio.size == output_audio.size)

if __name__ == "__main__":
    main()