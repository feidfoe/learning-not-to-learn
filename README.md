# Learning-Not-to-Learn
This repository is the official implementation([PyTorch](https://pytorch.org)) of https://arxiv.org/abs/1812.10352 which is published in CVPR2019.


## Conceptual Illustration
![teaser2](https://user-images.githubusercontent.com/45156153/64670700-f44c2e80-d4a0-11e9-9ac0-283d332a8941.PNG)

Since a neural network dfficiently learns data distribution, a network is likely to learn the bias information; the network can be as biased as the given data.

In the figure above, the points colored with high saturation indicate samples provided during training,
while the points with low saturation would appear in test scenario.
Although the classifier is well-trained to categorize the training
data, it performs poorly with test samples because the classifier
learns the latent bias in the training samples.

In this paper (and repo), we propose an iterative algorithm to unlearn the bias information.

## Requirements
1. NVIDIA docker : This code requires nvidia docker to run. If the nvidia docker is installed, the docker image will be automatically pulled. Other required libraries are installed in the docker image.

2. Pretrained model : If you don't want to use the pretrained parameters, erase the 'use_pretrain' and 'checkpoint' flags from the 'train.sh' script. They are trained without using the unlearning algoorithm. The checkpoint file can be found [here](https://drive.google.com/file/d/1mEpKquM8XAkaZXmyvtaszv49fjDp9Gd_/view?usp=sharing)

3. Dataset : Colored-MNIST dataset is constructed by the protocol proposed in https://arxiv.org/abs/1812.10352. They can be found [here](https://drive.google.com/file/d/11K-GmFD5cg3_KTtyBRkj9VBEnHl-hx_Q/view?usp=sharing). More details for the datasets are in dataset directory.

## Usage
First, download the pretrained model and dataset.
You can provide the directory for the dataset in 'option.py' (data_dir).

To train, modify the path to the pretrained checkpoint in train.sh.
Then, run the script with bash.
```
bash train.sh
```

For evaluation, run the test.sh script after modifying the paths.

```
bash test.sh
```

## Results
![confmat](https://user-images.githubusercontent.com/45156153/64670751-1776de00-d4a1-11e9-8f20-5898c2665e10.PNG)

Top row denotes the mean colors and their corresponding digit classes in training data. 
The confusion matrices of baseline model show the network is biased owing to the biased data. 
On the contrary, the networks trained by our algorithm are not biased to the color although they were trained with the same training data with the baseline


For comparison, we provide the experimental results with colored-MNIST. The table below is an alternative of Fig.4 in the paper.

|          |    0.02    |   0.025    |    0.03    |   0.035    |   0.04     |    0.045   |    0.05    |
| -------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Baseline |   0.4055   |   0.4813   |   0.5996   |   0.6626   |   0.7333   |   0.7973   |   0.8450   |
| BlineEye |   0.6741   |   0.7123   |   0.7883   |   0.8203   |   0.8638   |   0.8927   |   0.9159   |
|   Gray   |   0.8374   |   0.8751   |   0.8996   |   0.9166   |   0.9325   |   0.9472   |   0.9596   |
|   Ours   |   0.8185   |   0.8854   |   0.9137   |   0.9306   |   0.9406   |   0.9555   |   0.9618   |


## Notes
1. This is an example of unlearning using colored-MNIST data.

2. The main purpose of this code is to remove color information from extracted features.

3. Since this algorithm uses adversarial training, it is not very stable. 
In case you are suffering from the unstability, try pre-train f and g networks with h network detached, so the networks learn the bias. 
Then, take h network in the training loop (adversarial training).


### Contact
Byungju Kim(byungju.kim@kaist.ac.kr)

### BibTeX for Citation
```
@InProceedings{Kim_2019_CVPR,
author = {Kim, Byungju and Kim, Hyunwoo and Kim, Kyungsu and Kim, Sungjin and Kim, Junmo},
title = {Learning Not to Learn: Training Deep Neural Networks With Biased Data},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
