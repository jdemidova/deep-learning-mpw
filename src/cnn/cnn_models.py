import torch.nn as nn

from src.cnn.cnn_registry import register_model


# create a simple model with one convolutional layer and two fully connected layers
@register_model("improved_model")
class ImprovedModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, units=128, drop=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, units),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(units, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

@register_model("shallow_model")
class ShallowModel(nn.Module):

    def __init__(self, units=128, drop=0.5):
        super(ShallowModel, self).__init__()
        self.seq = nn.Sequential(
            # Layer 1---------------------------------------------------------------------------------------
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Conv with 32 filters, kernel size 3x3, padding 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224x224 -> 112x112

            # Layer 2---------------------------------------------------------------------------------------
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Conv with 64 filters, kernel size 3x3, padding 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112x112 -> 56x56

            # output layer----------------------------------------------------------------------------------
            nn.Flatten(),
            nn.Linear(56 * 56 * 64, units),
            nn.ReLU(),
            nn.Dropout(drop),  # dropout rate of 0.5 as default value
            nn.Linear(units, 10)  # output layer with 10 units for 10 classes
        )

    def forward(self, x):
        return self.seq(x)


# A convolutional block with optional pooling and dropout, can be reused to build deeper CNNs
@register_model("conv_block")
class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of Conv2d -> BatchNorm2d -> ReLU, with optional MaxPool2d and Dropout2d.
    """
    def __init__(self, in_channels: int, out_channels: int, use_pool: bool = False, dropout: float = 0.0):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False), # conv with kernel size 3x3, padding 'same', no bias (since we have batch norm)
            nn.BatchNorm2d(out_channels), # batch normalization for better training stability
            nn.ReLU(inplace=True), # ReLU activation with inplace=True for memory efficiency
        ]

        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) # max pooling with kernel size 2x2 and stride 2 to downsample the feature maps

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout)) # spatial dropout to randomly zero out entire channels during training for regularization

        self.block = nn.Sequential(*layers) # sequential container to hold the layers of the block

    def forward(self, x):
        return self.block(x)



# Construction of the main CNN model with variable depth using the ConvBlock defined above
@register_model("depth_cnn")
class DepthCNN(nn.Module):
    """
    CNN with configurable number of convolutional layers.
    Supported depths: 1, 2, 4, 8, 16
    """

    def __init__(
        self,
        depth: int,
        in_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 32, # number of filters in the first conv layer, will be doubled after every second conv layer
        max_channels: int = 256, # maximum number of filters to prevent excessive model size
        dropout: float = 0.0, # dropout rate set to zero after discussion with Jean.
                              # He has found that dropout does not help in this setting and just increases training time, so we will set it to zero by default
        do: float = 0.5, # dropout rate for the final classifier head
        inputsize: int = 224, # input image size, can be used to calculate the size of the feature maps after the convolutional layers for the classifier head
    ):
        super().__init__()

        if depth not in {1, 2, 4, 8, 16, 32}: # only support specific depths to keep the model sizes reasonable and comparable
            raise ValueError(f"Depth must be one of {{1,2,4,8,16,32}}, got {depth}")

        layers = []

        current_in = in_channels
        current_out = base_channels
        current_size = inputsize

        pool_count = 0 # to keep track of how many times we have pooled, since we pool after every second conv layer

        # build the convolutional part of the model by stacking ConvBlocks according to the specified depth, doubling the number of filters after every second block until max_channels is reached
        for i in range(depth):
            want_pool = ((i + 1) % 2 == 0) # pool after every second conv layer common practice to reduce spatial dimensions while increasing feature channels

            # "only pool if spatial size is still >= 2"
            use_pool = want_pool and current_size >= 2  and pool_count < 4 # limit the number of pooling layers to prevent reducing spatial dimensions too much, since we start with 224x224, after 4 pools we would be at 14x14 which is still reasonable for the classifier head

            layers.append(
                ConvBlock(
                    in_channels=current_in, # number of input channels for this block, will be the output channels of the previous block
                    out_channels=current_out, # number of output channels for this block, will be doubled after every second block until max_channels is reached
                    use_pool=use_pool, # whether to apply max pooling in this block, applied after every second conv layer see above
                    dropout=dropout if depth >= 4 else 0.0, # is set to zero see above
                )
            )

            current_in = current_out # for the next block, the input channels will be the output channels of the current block

            if use_pool:
                pool_count += 1
                current_out = min(current_out * 2, max_channels)

        self.features = nn.Sequential(*layers) # sequential container to hold all the convolutional blocks as the feature extractor part of the model

        # Final classifier head that takes the output of the convolutional part and produces class logits, using adaptive average pooling to handle variable spatial dimensions and a fully connected layer to produce the final class scores
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # global average pooling to reduce the spatial dimensions to 1x1, resulting in a feature vector of size current_in
            nn.Flatten(),
            nn.Dropout(p=do), # dropout before the final linear layer for regularization
            nn.Linear(current_in, num_classes)
        )

    # forward pass through the model: first through the convolutional feature extractor, then through the classifier head to produce the final logits
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
