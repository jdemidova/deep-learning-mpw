import os
import wandb

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
from torchvision import datasets, transforms

# =========================================================
# LOADING DATA 
# =========================================================

def get_dataset(imagesize=224):
    """Load the training and validation datasets for image classification.\n
    Args:
        imagesize (int): The desired size to which all images will be resized. Default is 224.
        Can be set to 128 or 64 for faster training but may reduce accuracy.
    Returns:
        train_dataset (torchvision.datasets.ImageFolder): The training dataset.
        val_dataset (torchvision.datasets.ImageFolder): The validation dataset.
    """
    # Original image size 224x224, can be changed to 128x128 or 64x64 for faster training but may reduce accuracy
    size = imagesize

    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        root='../data/icosimal_img_class_03/train',
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root='../data/icosimal_img_class_03/validate',
        transform=val_transform
    )

    return train_dataset, val_dataset



# =========================================================
# CONFIG
# =========================================================

@dataclass
# creating a dataclass to hold all configuration parameters for easy management and logging with default values
class ModelConfig:
    # data / model
    in_channels: int = 3 # RGB input
    num_classes: int = 10 # 10 classes in the dataset
    inputsize: int = 224 # can be set to 128 or 64 for faster training but may reduce accuracy
    depth: int = 12 # number of convolutional layers, can be set to 4 or 8 for faster training but may reduce accuracy
    base_channels: int = 32 # number of channels in the first conv layer, will be doubled after each pooling layer up to 256
    max_channels: int = 256 # maximum number of channels in conv layers, prevents excessive memory usage in deeper models
    dropout_conv: float = 0.0 # dropout rate set to zero after discussion with Jean.
                              # He has found that dropout does not help in this setting and just increases training time, so we will set it to zero by default
    dropout_fc: float = 0.5 # dropout rate for the final classifier head

    # training
    lr: float = 1e-3 # learning rate of the optimizer, can be tuned for better performance
    batch_size: int = 64 # default size of mini-batches for training
    epochs: int = 20 # default value 
    num_workers: int = 0 # number of worker processes for data loading, set to 0 for compatibility with Windows and to avoid potential issues in some environments. Can be increased to speed up data loading if supported.
    pin_memory: bool = True # whether to use pinned memory for data loading, can improve performance when using CUDA but may cause issues in some environments, so set to True by default but can be disabled if needed
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # use GPU if available, otherwise fall back to CPU

@dataclass
# wandb default settings for wandb logging
class WandbConfig:
    use_wandb: bool = True
    project: str = "MPW-CNN"
    entity: str = "MSE_DeLearn_SPR26"
    mode: str = "online"      # "online", "offline", "disabled"
    login: bool = True
    api_key: Optional[str] = None
    init_timeout: int = 300
    watch: bool = False
    log_freq: int = 100


# =========================================================
# MODEL
# =========================================================

# A convolutional block with optional pooling and dropout, can be reused to build deeper CNNs
class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of Conv2d -> BatchNorm2d -> ReLU, with optional MaxPool2d and Dropout2d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_pool: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False), # conv with kernel size 3x3, padding 'same', no bias (since we have batch norm)
            nn.BatchNorm2d(out_channels), # batch normalization for better training stability and performance
            nn.ReLU(inplace=True), # ReLU activation function
        ]

        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # max pooling with kernel size 2x2 and stride 2 to downsample the feature maps

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout)) # spatial dropout to randomly zero out entire channels during training for regularization

        self.block = nn.Sequential(*layers) # sequential container to hold the layers of the block

    def forward(self, x: torch.Tensor):
        return self.block(x)


# Construction of the main CNN model with variable depth using the ConvBlock defined above
class DepthCNN(nn.Module):
    """
    CNN with configurable number of convolutional layers.
    """
    def __init__(
        self,
        depth: int,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 32,
        max_channels: int = 256,
        dropout_conv: float = 0.0,
        dropout_fc: float = 0.5,
        inputsize: int = 224,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers = []
        current_in = in_channels
        current_out = base_channels
        current_size = inputsize
        pool_count = 0 # to keep track of how many times we have pooled, since we pool after every second conv layer

        # build the convolutional part of the model by stacking ConvBlocks according to the specified depth, doubling the number of filters after every second block until max_channels is reached
        for i in range(depth):
            want_pool = ((i + 1) % 2 == 0) # pool after every second conv layer common practice to reduce spatial dimensions while increasing feature channels

            # do not pool below valid spatial size, and cap total number of pools
            # limit the number of pooling layers to prevent reducing spatial dimensions too much, since we start with 224x224, after 4 pools we would be at 14x14 which is still reasonable for the classifier head
            use_pool = want_pool and current_size >= 2 and pool_count < 4 

            layers.append(
                ConvBlock(
                    in_channels=current_in,
                    out_channels=current_out,
                    use_pool=use_pool,
                    dropout=dropout_conv if depth >= 4 else 0.0,
                )
            )

            current_in = current_out # for the next block, the input channels will be the output channels of the current block

            if use_pool:
                pool_count += 1
                current_size = current_size // 2 # after pooling, the spatial dimensions are halved
                current_out = min(current_out * 2, max_channels)

        self.features = nn.Sequential(*layers)

        # Final classifier head that takes the output of the convolutional part and produces class logits, using adaptive average pooling to handle variable spatial dimensions and a fully connected layer to produce the final class scores
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # global average pooling to reduce the spatial dimensions to 1x1, resulting in a feature vector of size current_in
            nn.Flatten(),
            nn.Dropout(p=dropout_fc), # dropout before the final linear layer for regularization
            nn.Linear(current_in, num_classes),
        )

    # forward pass through the model: first through the convolutional feature extractor, then through the classifier head to produce the final logits
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

# helper function to build the model from the configuration parameters, makes it easier to manage and log the model architecture based on the config
def build_model(cfg: ModelConfig) -> nn.Module:
    return DepthCNN(
        depth=cfg.depth,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
        max_channels=cfg.max_channels,
        dropout_conv=cfg.dropout_conv,
        dropout_fc=cfg.dropout_fc,
        inputsize=cfg.inputsize,
    )

# count the number of trainable parameters for logging with wandb
def get_num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =========================================================
# DEVICE
# =========================================================

# helper funktion to work in GPU if available, other wise use CPU
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# W&B
# =========================================================

# helper function for wandb login
def wandb_login_if_needed(cfg: WandbConfig) -> None:
    """
    Performs W&B login once if enabled.

    Priority:
    1. cfg.wandb_api_key
    2. environment variable WANDB_API_KEY
    3. existing local wandb login state
    """
    if not cfg.use_wandb or cfg.mode == "disabled":
        return

    if not cfg.login:
        return

    api_key = cfg.api_key or os.getenv("WANDB_API_KEY")

    try:
        if api_key:
            wandb.login(key=api_key, relogin=False)
        else:
            # uses saved login if available, otherwise may prompt in terminal
            wandb.login(relogin=False)
    except Exception as e:
        raise RuntimeError(
            f"W&B login failed. Check your API key or login state. Original error: {e}"
        ) from e

# initialise wandb run
def init_wandb_run(
    cfg: WandbConfig,
    model: Optional[nn.Module] = None,
    run_name: Optional[str] = None,
    config_dict: Optional[dict] = None,
):
    if not cfg.use_wandb or cfg.mode == "disabled":
        return None

    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        mode=cfg.mode,
        name=run_name,
        config=config_dict,
        reinit=True,
        settings=wandb.Settings(init_timeout=cfg.init_timeout),
    )

    if model is not None and cfg.watch:
        wandb.watch(model, log="all", log_freq=cfg.log_freq)

    return run