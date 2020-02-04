AI Programming with Python Project
==================================

Project code for Udacity's AI Programming with Python Nanodegree program. 

The aim of the project is to make an image classifier to recognize a dataset of flowers. There are 102 types of flowers.

You can either download the dataset from [this link](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) or use first_time_setup.py.

The dataset has 3 sets of data for training, validation, and testing.

There are 3 python scripts that run from the command line. Usage examples are as follows:

first_time_setup.py
-------------------

*Downloads and extracts dataset to the 'flowers' directory*

python first_time_setup.py

python first_time_setup.py --save_dir save (data set will be downloaded and extracted to the 'save' directory)

train.py
--------

*trains and validates the classifier and saves the checkpoint to the specified directory*

python train.py (data set shall be initially extracted to the 'flowers' directory)

python train.py data_dir (data set shall be initially extracted to the 'data_dir' directory)

python train.py data_dir --save_dir save_directory (set directory to save checkpoints)

python train.py data_dir --arch "vgg13" (choose architecture from vgg13, vgg16 and vgg19)

python train.py data_dir --learning_rate 0.01 --hidden_units [1024, 512, 256] --epochs 20 (set hyperparameters)

predict.py
----------

*predicts an image using specified checkpoint and returns top K most likely classes*

python predict.py ( use default image 'flowers/test/1/image_06743.jpg' and root directory for checkpoint)

python predict.py /path/to/image checkpoint (predict the image in /path/to/image using checkpoint)

python predict.py --top_k 3 (return top K most likely classes)

python predict.py --category_names cat_to_name.json (use a mapping of categories to real names)

python predict.py --gpu (use GPU for inference)