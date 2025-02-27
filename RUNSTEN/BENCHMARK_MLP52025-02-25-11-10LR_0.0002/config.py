from datetime import datetime

BATCH_SIZE = 8
TEST_PERCENT = 0.2
PATTERNS_FILE = "patterns.pt"
LEARNING_RATE = 2e-4
EPOCHS = 60
SIZE_PATTERNS = 9 # DISPARI
POINTS_PER_IMAGE = 1
NUM_IMAGES = 120
NUM_CLASSES = 10
RUN_NAME = "BENCHMARK_MLP5"+datetime.now().strftime('%Y-%m-%d-%H-%M')+"LR_"+str(LEARNING_RATE)
# RUN_NAME = datetime.now().strftime('%Y-%m-%d-%H-%M')+"LR_"+str(LEARNING_RATE)+"-SzPtt_"+str(SIZE_PATTERNS)+"-NImg_"+str(NUM_IMAGES)+"-PPImg_"+str(POINTS_PER_IMAGE)