# Super Resolution Audio Stretch (SRAS)
### A deep-learned time-stretch algorithm for audio

Training data consists of MIDI files rendered by a string emulation synthesizer. The output audio files for training are created by playing the MIDI file at half the BPM as the input. Thus, the input audio files are half the length of the output files.

I am currently experimenting with different architectures for the neural net and will update this README when I've achieved better results.

### TO-DO:
~~- Spectrogram conversion/data handling~~
~~- Spectrogram pre-upscale model~~
- Spectrogram post-upscale model
- SR-GAN using audio spectrograms
