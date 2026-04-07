import torch.nn as nn
import torch

from src.cnn.cnn_registry import register_model

# create a simple model with one convolutional layer and two fully connected layers
@register_model("first_model")
class first_model(nn.Module):
    
    def __init__(self, units=128):
        super(first_model, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Conv with 32 filters, kernel size 3x3, padding 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(112*112*32,units),
            nn.ReLU(),
            nn.Linear(units,10) # output layer with 10 units for 10 classes
        )
        
    
    def forward(self, x):
        return self.seq(x)


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


@register_model("improved_model_no_lin")
class improved_model(nn.Module):
    
    def __init__(self, drop=0.5):
        super(improved_model, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding='same'), # Conv with 32 filters, kernel size 3x3 and padding 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), # global average pooling to reduce the spatial dimensions to 1x1, resulting in a feature vector of size 32
            nn.Flatten(),
            nn.Dropout(drop), # dropout rate of 0.5 as deafault value
            nn.Linear(32,10) # output layer with 10 units for 10 classes
        )
        
    
    def forward(self, x):
        return self.seq(x)

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

# =========================================================
# Building blocks
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


@register_model("depth_cnn")
# Construction of the main CNN model with variable depth using the ConvBlock defined above
class DepthCNN(nn.Module):
    """
    CNN with configurable number of convolutional layers.
    """
    def __init__(
        self,
        depth: int = 12,
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
    

@register_model("depth_cnn_fc")
class DepthCNNFC(nn.Module):
    """
    DepthCNN with configurable number of convolutional layers and
    fully connected layers in the classifier head.

    fc_layers:
    1 -> features -> num_classes
    2 -> features -> hidden1 -> num_classes
    3 -> features -> hidden1 -> hidden2 -> num_classes
    4 -> features -> hidden1 -> hidden2 -> hidden3 -> num_classes
    5 -> features -> hidden1 -> hidden2 -> hidden3 -> hidden4 -> num_classes
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
        fc_layers: int = 1,
        hidden1: int = 128,
        hidden2: int = 64,
        hidden3: int = 32,
        hidden4: int = 16,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")
        if fc_layers < 1 or fc_layers > 5:
            raise ValueError("fc_layers must be 1, 2, 3, 4, or 5")

        layers = []
        current_in = in_channels
        current_out = base_channels
        current_size = inputsize
        pool_count = 0

        for i in range(depth):
            want_pool = ((i + 1) % 2 == 0)
            use_pool = want_pool and current_size >= 2 and pool_count < 4

            layers.append(
                ConvBlock(
                    in_channels=current_in,
                    out_channels=current_out,
                    use_pool=use_pool,
                    dropout=dropout_conv if depth >= 4 else 0.0,
                )
            )

            current_in = current_out

            if use_pool:
                pool_count += 1
                current_size = current_size // 2
                current_out = min(current_out * 2, max_channels)

        self.features = nn.Sequential(*layers)

        classifier_layers = [
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ]

        # hard coded fully connected layers based on the fc_layers parameter, with ReLU activations and dropout in between for regularization, allowing for more complex classifier heads if desired
        if fc_layers == 1:
            classifier_layers.extend([
                nn.Dropout(p=dropout_fc),
                nn.Linear(current_in, num_classes),
            ])

        elif fc_layers == 2:
            classifier_layers.extend([
                nn.Linear(current_in, hidden1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden1, num_classes),
            ])

        elif fc_layers == 3:
            classifier_layers.extend([
                nn.Linear(current_in, hidden1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden2, num_classes),
            ])

        elif fc_layers == 4:
            classifier_layers.extend([
                nn.Linear(current_in, hidden1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden2, hidden3),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden3, num_classes),
            ])

        elif fc_layers == 5:
            classifier_layers.extend([
                nn.Linear(current_in, hidden1),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden2, hidden3),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden3, hidden4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_fc),
                nn.Linear(hidden4, num_classes),
            ])
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x