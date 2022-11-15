# Jupyter Notebooks
The notebooks documenting the data collection, data processing and models are in the `notebook` folder.
1. [Data Collection](notebooks/01_Data_Collection.ipynb)
1. [Data Exploration and Processing](notebooks/02_DataExploration_PreProcessing.ipynb)
1. [Baseline CNN and Improvements](notebooks/03_Baseline_CNN_Model_and_Improvements.ipynb)
    1. Baseline model
    1. Improvements to baseline model
    1. Residual models
1. [Exploration of Other Networks](notebooks/04_Product_Classification_Other_Networks.ipynb)
    1. RNN+CNN model
    1. CNN+Attention models
        1. Local attention
        1. Convolution self attention

# Data
The data used to train the model can be downloaded [here](https://drive.google.com/drive/folders/1uqZkgFSJA2R8oVQZQtdou54A0no3g7Nj) 

# Dependencies
python: 3.9
### Data analysis and processing
scikit-image: 0.19.3
scipy: 1.9.3
opencv-python: 4.6.0.66
torchvision; 0.13.1
torchtext: 0.13.1
imagehash: 4.2.1
matplotlib: 3.6.0
pandas: 1.4.3
rembg

### Model training
numpy: 1.23.4
matplotlib: 3.6.0
opencv-python: 4.6.0.66
torch: >=1.12.1
torchvision; 0.13.1
tqdm
