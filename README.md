
# Machine Learning for Flower Classification

## Overview
Welcome to the Flower Classification project! This project leverages machine learning techniques to classify images of flowers into various species. The primary goal is to build a robust model capable of accurately identifying different types of flowers.

## Key Features
- Training Script: Utilizes the train.py script to train a machine learning model on a labeled dataset of flowers.
- Prediction Script: The predict.py script allows users to make predictions on new images using a pre-trained model.
- First-Time Setup: The first_time_setup.py script assists in downloading and preparing the dataset for training.

## Dataset
The project uses the Udacity Flower dataset, containing a variety of flower species. The dataset is structured to facilitate model training, validation, and testing.

## Model Architecture
The project employs the VGG (Visual Geometry Group) architecture, specifically VGG-16, as the backbone for the flower classification model. The model is trained to recognize distinct features of different flower species.

## Technical Requirements
To ensure optimal functionality of this project, the following technical setup is recommended:

- **Python Version**: Python 3.7.3. The project is developed and tested using this specific Python version for maximum compatibility.

- **Libraries and Dependencies**:
  - This project relies on several key Python libraries, with specific versions listed in the `requirements.txt` file. This includes:
    - **PyTorch 1.3.1**: For deep learning functionalities.
    - **NumPy 1.16.2** and **Pandas 0.24.2**: For numerical operations, data handling, and preprocessing.
    - **Pillow 5.4.1**: For image processing and manipulation.
  - To install these dependencies, use the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

- **Hardware Requirements**: Although the project can run on a CPU, a GPU is recommended for efficient training of deep learning models, particularly for complex or large datasets.

- **Operating System**: The project is compatible with major operating systems such as Windows, macOS, and Linux. Note that installation steps for Python and its libraries might vary slightly across different platforms.


## Installation

Follow these steps to set up the environment and run the "Machine Learning for Flower Classification" project:

1. **Clone the Repository**:
   Begin by cloning the repository to your local machine. Use the command:
   git clone https://github.com/ErkanHatipoglu/AIPND_final_project_part_2.git
   cd AIPND_final_project_part_2

2. **Set Up Python Environment**:
   It's recommended to create a new Python environment for this project. You can use Conda or any other virtual environment manager. For Conda, use:
   conda create --name flower-classification python=3.7.3
   conda activate flower-classification

3. **Install Required Packages**:
   Install all the required packages using the `requirements.txt` file. Run:
   pip install -r requirements.txt

4. **Verify Installation**:
   After the installation, verify that all the necessary packages are correctly installed by running:
   pip list

5. **Run the Project**:
   Once everything is set up, you are ready to run the project scripts like `train.py` or `predict.py`.


## Usage

To use the "Machine Learning for Flower Classification" project, follow these steps:

1. **Getting the Data**:
   To train the model, You can either download the dataset from [this link](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) or navigate to the project directory and run the `first_time_setup.py` script. This script downloads and extracts dataset to the 'flowers' directory. Example:

   python first_time_setup.py

   python first_time_setup.py --save_dir save (data set will be downloaded and extracted to the 'save' directory)

2. **Training the Model**:
   To train the model, navigate to the project directory and run the `train.py` script. You can specify the dataset and other parameters as needed. Example:
   
   python train.py (data set shall be initially extracted to the 'flowers' directory)

   python train.py /path/to/flower_data (data set shall be initially extracted to the 'data_dir' directory)

   python train.py /path/to/flower_data --save_dir save_directory (set directory to save checkpoints)

   python train.py /path/to/flower_data --arch "vgg13" (choose architecture from vgg13, vgg16 and vgg19)

   python train.py /path/to/flower_data --learning_rate 0.01 --hidden_units [1024, 512, 256] --epochs 20 (set hyperparameters)

3. **Using the Trained Model for Prediction**:
   After training, use the `predict.py` script to classify new images. Provide the path to the image and the trained model checkpoint. Example:
   python predict.py /path/to/image.jpg checkpoint.pth

   python predict.py ( use default image 'flowers/test/1/image_06743.jpg' and root directory for checkpoint)

   python predict.py /path/to/image checkpoint (predict the image in /path/to/image using checkpoint)

   python predict.py --top_k 3 (return top K most likely classes)

   python predict.py --category_names cat_to_name.json (use a mapping of categories to real names)

   python predict.py --gpu (use GPU for inference)

4. **Customizing Parameters**:
   Both training and prediction scripts can be customized with additional command-line arguments such as setting hyperparameters for training or choosing the top K classes for prediction.

5. **Reviewing Results**:
   The training script will output the model's performance metrics, and the prediction script will display the top predicted classes along with associated probabilities.

Ensure to replace `/path/to/flower_data`, `/path/to/image.jpg`, and `checkpoint.pth` with the actual paths in your environment.

```

## Acknowledgments
Credits for any third-party resources or collaborators.
