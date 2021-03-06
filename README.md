# Scene Segmentation

## Training and Testing Pipeline

This codebase contains training and testing modules developed for scene segmentation.

### Dataset

The dataset used in this pipeline can be downloaded **[here](https://drive.google.com/file/d/1oZSOkd4lFmbY205VKQ9aPv1Hz3T_-N6e/view?usp=sharing)**. Please make sure to store the dataset inside the `dataset` directory. The partition of the dataset can be found inside `data/train.txt` and `data/test.txt`.

### Weights

The training weights can be found on this **[link](https://drive.google.com/file/d/1oZSOkd4lFmbY205VKQ9aPv1Hz3T_-N6e/view?usp=sharing)**. The default location of weights is inside the `weights` directory. 

Samples and epochs were reduced due to limited training time. An mAP of 65% and a mean miou of 64.6% were obtained when the model was evaluated.

## Training

Run the training using the following command:

```
python train.py --mode train
```

To evaluate the model, run:

```
python test.py --mode train
```

The metrics used in this codebase is obtained from this **[repository](https://github.com/eluv-io/elv-ml-challenge)**.