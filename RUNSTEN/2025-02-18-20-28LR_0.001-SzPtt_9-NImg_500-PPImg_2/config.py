from datetime import datetime

BATCH_SIZE = 8
TEST_PERCENT = 0.2
PATTERNS_FILE = "patterns.pt"
LEARNING_RATE = 1e-3
EPOCHS = 10
SIZE_PATTERNS = 9 # DISPARI
POINTS_PER_IMAGE = 2
NUM_IMAGES = 500
NUM_CLASSES = 10
# RUN_NAME = "MODEL_FIDELITY2"
RUN_NAME = datetime.now().strftime('%Y-%m-%d-%H-%M')+"LR_"+str(LEARNING_RATE)+"-SzPtt_"+str(SIZE_PATTERNS)+"-NImg_"+str(NUM_IMAGES)+"-PPImg_"+str(POINTS_PER_IMAGE)