# Building an Image Classifier with TFLearn

##### Introduction
Image classification has been a very fascinating task for computers for a very long time. Whilst a human can easily identify objects in an image, it is difficult for computers to do so. The human brain _automagically_ understands what it is looking at given inputs of vision from the eyes, from what it has learnt from experience. The computer analyzes images from the pixels of the given image by comparing it to what it has seen before.

Assuming you have already installed TFLearn and its dependencies, in this tutorial, you will learn how to:

- Prepare images for processing (without preprocessed data).
- Program and train a neural network with the images.
- Test your model on an image.

##### Architecture
So how can we teach the computer to classify pictures of various objects? The answer is a special type of neural network called a **convolutional neural network**. Many researchers have devised many variations of this neural network, which all differ in hyperparameters such as number of nodes, layers, etc, so there is no set solution for each task. The design process for such networks are usually empirical with a lot of experimentation that all depend on factors such as computing power and data.
##### Preprocessing
For our example, we will construct a neural network to distinguish cats from dogs using data that is provided [here](https://www.kaggle.com/c/dogs-vs-cats/data).
Download the `train.zip` file and extract all the images into a single directory.

We will now whip up a small Python script that will help us split our data into two sections: `training` and `cross-validation`, with 70% and 30% of the original set, respectively. This script is useful for this specific dataset. When utilizing your own images, your mileage may vary.

You may also need to install `h5py` as a dependency as we are using a specific dataset format.

```python
# Run this in the directory above your images.
import os

path_to_training = 'train_data.txt'
path_to_cval = 'cval_data.txt'

cval = open(path_to_cval, 'a')
training = open(path_to_training, 'a')

for (path, dirnames, filenames) in os.walk('NAME_OF_YOUR_IMAGE_DIRECTORY'):

    i = 0.7 * len(filenames)
    for name in filenames[:int(i)]:
        if 'dog' in name:
            training.write(os.path.join(path, name) + ' 1\n')
        else:
            training.write(os.path.join(path, name) + ' 0\n')
    for name in filenames[int(i):]:
        if 'dog' in name:
            cval.write(os.path.join(path, name) + ' 1\n')
        else:
            cval.write(os.path.join(path, name) + ' 0\n')

from tflearn.data_utils import build_hdf5_image_dataset

build_hdf5_image_dataset(path_to_training, image_shape=(32, 32, 3), mode='file', output_path='training_data.h5', categorical_labels=True, normalize=True)
build_hdf5_image_dataset(path_to_cval, image_shape=(32, 32, 3), mode='file', output_path='cval_data.h5', categorical_labels=True, normalize=True)

```

The magic happens in the function `build_hdf5_image_dataset`, which is a useful helper in TFLearn that helps process all your images into a single dataset. More information can be found in the official [documentation](http://tflearn.org/data_utils/#build-hdf5-image-dataset). We've packed our images to be 32x32 pixels wide with RGB values as well.

After running this script, you should have the files `training_data.h5` and `cval_data.h5`.

Now that we have our processed datasets, we can now continue on to construct the neural network.

##### Training
