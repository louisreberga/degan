import torch

START_TRAIN_AT_IMG_SIZE = 4
DATASET = ""
SAVE_PATH = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [32, 32, 32, 16, 16, 16, 16, 8, 4]
CHANNELS_IMG = 3
Z_DIM = 512  # 512 in the paper
IN_CHANNELS = 512  # 512 in the paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [50] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(3, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 2

