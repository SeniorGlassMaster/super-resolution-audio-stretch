MODEL = 'pre_s'
# MODEL = 'post'

LOAD_MODEL = False
LOAD_MODEL_PATH = "./model_saves/pre_s_model_0_e999.pth"

LIVE_GRAPH = False

# Non-spectrogram settings:
WINDOW_SIZE = 1000
OVERLAP = 50

# Spectrogram settings:
NPERSEG = 512
BATCHES_PER_EPOCH = 100
FRAMES_PER_BATCH = 3

NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4

MODEL_NAME = "pre_s_model_0"
SAVE_PATH = "./model_saves/" + MODEL_NAME
EXPORT_PATH = "./output/" + MODEL_NAME + ".wav"
