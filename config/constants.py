PSYCHOLOGICAL_STATES = ["calm", "stressed", "anxious", "confident", "frustrated"]
NUM_CLASSES = len(PSYCHOLOGICAL_STATES)
STATE_TO_IDX = {state: idx for idx, state in enumerate(PSYCHOLOGICAL_STATES)}
IDX_TO_STATE = {idx: state for idx, state in enumerate(PSYCHOLOGICAL_STATES)}

NUM_POSE_LANDMARKS = 33
NUM_FACE_LANDMARKS = 478
POSE_FEATURE_DIM = NUM_POSE_LANDMARKS * 3  # x, y, z

IMAGE_SIZE = 224
SAMPLE_RATE = 16000
N_MFCC = 40
