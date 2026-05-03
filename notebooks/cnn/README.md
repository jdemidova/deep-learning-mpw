# 1 - CNN Task
* Deadline: Tue, 14 April 15:00
* Submission (report) format: PDF or Jupyter NB

## Goal
The goal is to **apply the concepts** seen in the lectures and practical sessions, 
and to **document your experimentation process and results** in a report. 

## Requirements on report
- _Clear, concise, and well-structured_,
- Includes all _relevant details about your experiments_: 
  - the **_hyperparameters_** you used, 
  - the **_results_** you obtained,
  - any **_insights or conclusions_** you can draw from your experiments.
- **_States group members_** in the beginning of the report.

## Objectives
### Mandatory objectives
1. Start from **simple CNN** architectures and **progressively increase their complexity** to show
the benefit of depth.
2. Show the **importance of hyperparameter tuning**.
3. Show models that **overfit and underfit** and explain the reasons for that.
4. Experiment with **regularization techniques**.
5. Experiment with **different optimization algorithms** (e.g., Adam, RMSprop) and **compare**
their performance.
#### Hint 
Resizing the images e.g., 128x128 or 64x64 may be useful at the beginning of your
investigations to perform quicker experimentation and sweep towards good settings.

### Optional objectives
Pick **at least 2** of the following optional objectives to further explore and analyze the performance
of your models :

- Perform **error analysis** to identify common misclassifications.
- **Manually inspect misclassifications** of your best model to identify mis-labelled samples
in the dataset. Analyze their impact on model performance (i.e. when they are removed
from the training set, when they are removed from the validation set, when they are kept
in both sets). If you do so, provide logs of the mis-labelled images.
- Use **tools to monitor the training** process and analyze the training and validation curves
(e.g., Tensorboard, Weights & Biases).
- Use frameworks to automatically **search for the best hyperparameters** (e.g., Optuna, Hyperopt)
and compare their performance with manual tuning.
- Use **advanced architectures** (e.g., ResNet, DenseNet) and compare their performance with
simpler architectures.
- Perform **transfer learning** using pre-trained models and compare their performance with
models trained from scratch.
- Experiment with **data augmentation techniques** to improve model generalization (e.g.,
random cropping, horizontal flipping, color jittering).
- **Augment the dataset** by re-collecting images (e.g., using the Bing API with imagescraper.
py) ; use your best model to validate the newly collected images.
- Use **visualization techniques** (e.g., Grad-CAM) to understand which parts of the images
the model is focusing on when making predictions.

