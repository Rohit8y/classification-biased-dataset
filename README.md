# classification-biased-dataset
Challenge: The training datatset has 4 classes of mugs: (White, Black-Label, Blue, Transparent) with 500 images in each class. The objective is to identify the 2nd
class: black mug with company label. Although the dataset has been biased intentionally by adding multiple class images in the 2nd class training dataset.

Goal: Classify the images to get the maximum probability for the 2nd class images.

Data:To download the data execute the following command (you will need to install the gsutil command beforehand which is part of the Google Cloud SDK):
```console
gsutil -m cp -R gs://ml6_junior_ml_engineer_challenge_cv_mugs_data/data .
```

![alt text](data.png "Title")

