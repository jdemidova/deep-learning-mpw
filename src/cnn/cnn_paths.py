from pathlib import Path
from paths_global import CNN_DIR

CNN_DIR = Path(CNN_DIR)

DATA_DIR = CNN_DIR / "data"
DATASET_DIR = DATA_DIR / "icosimal_img_class_03"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validate"

SPLIT_DIR = DATA_DIR / "split"