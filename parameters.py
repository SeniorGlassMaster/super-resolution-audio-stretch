MODEL = 'pre'
# MODEL = 'post'

LOAD_MODEL = True
LOAD_MODEL_PATH = "./model_saves/pre_model_0_e1.pth"

WINDOW_SIZE = 1000
OVERLAP = 50
NUM_EPOCHS = 20
LEARNING_RATE = 1e-6
USE_CUDA = False

MODEL_NAME = "pre_model_0"
SAVE_PATH = "./model_saves/" + MODEL_NAME
EXPORT_PATH = "./output/" + MODEL_NAME + ".wav"\