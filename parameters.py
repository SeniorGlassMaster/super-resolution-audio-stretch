MODEL = 'pre_s'
# MODEL = 'post'

LOAD_MODEL = False
LOAD_MODEL_PATH = "./model_saves/pre_s_model_0_e5.pth"

LIVE_GRAPH = False

WINDOW_SIZE = 1000
OVERLAP = 50
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

MODEL_NAME = "pre_s_model_0"
SAVE_PATH = "./model_saves/" + MODEL_NAME
EXPORT_PATH = "./output/" + MODEL_NAME + ".wav"
