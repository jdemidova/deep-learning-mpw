# Datasets

## CNN task
We will use the **iCoSimal V3 dataset** that contains **30’000 images of animals in 10 categories**.
It has the following characteristics :
- Number of classes : 10
- Number of images : 30’000
- Image resolution : 224x224 pixels
- **Split** : 24’000 training images, 6’000 ’validation’ images (to be used as the test set)
- Class distribution : Balanced
- Classes : cat, chicken, cow, dog, elephant, horse, rabbit, sheep, squirrel, zebra
- Source of images : Collected from direct downloads from the net (using Bing API) and
various open-source image repositories including Coco and OpenImages v7 datasets.
- Grooming : to avoid duplicates and prepare the sets, image-groomer.py was used.