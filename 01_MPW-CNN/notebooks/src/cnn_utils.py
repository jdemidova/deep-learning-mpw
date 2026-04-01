from torchvision import datasets, transforms

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