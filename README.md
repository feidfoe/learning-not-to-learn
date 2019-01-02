# learning-not-to-learn
This repository contains [PyTorch](https://pytorch.org) implementation for https://arxiv.org/abs/1812.10352.

## Requirements
1. NVIDIA docker : This code requires nvidia docker to run. If the nvidia docker is installed, the docker image will be automatically pulled. Other required libraries are installed in the docker image.

2. Pretrained model : If you don't want to use the pretrained parameters, erase the 'use_pretrain' and 'checkpoint' flags from the 'train.sh' script. They are trained without using the unlearning algoorithm. The checkpoint file can be found [here](https://drive.google.com/file/d/1mEpKquM8XAkaZXmyvtaszv49fjDp9Gd_/view?usp=sharing)

3. Dataset : Colored-MNIST dataset is constructed by the protocol proposed in https://arxiv.org/abs/1812.10352. They can be found [here](https://drive.google.com/file/d/11K-GmFD5cg3_KTtyBRkj9VBEnHl-hx_Q/view?usp=sharing)

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


## Notes
1. This is an example of unlearning using colored-MNIST data.

2. The main purpose of this code is to remove color information from extracted features.


### Contact
Byungju Kim(byungju.kim@kaist.ac.kr)
