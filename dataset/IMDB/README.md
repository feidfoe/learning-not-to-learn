# IMDB dataset with bias

IMDB face images can be downloaded [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).
Filtered information we have used for our experiments can be downloaded [here](https://drive.google.com/file/d/1ZFZ2tUjq3BBRw3rcXB0LkBvaqKD4_8WK/view?usp=sharing).

imdb_age_gender.npy contains age and gender information for each images

imdb_split.npy contains train and test split we have used.

You can load the data with following snippet:
```
gt = np.load('imdb_age_gender.npy', encoding='latin1').item()
sp = np.load('imdb_split.npy', encoding='latin1').item()
image_list = dic['train_list']  # depends on whether train or test (train_list/test_list)
labels = gt[image_list[i].encode('utf-8')]
age = int(labels['age'])
gender = int(labels['gender'])

```

