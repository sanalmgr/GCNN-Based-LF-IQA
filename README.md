# GCNN-Based-LF-IQA
In this work, we propose a no-reference LF-IQA method that predicts the quality of compressed LF images using a Deep Graph Convolutional Neural Network (GCNN-LFIQA). The GCNN-LFIQA method is based on a deep single-stream network architecture which takes horizontal EPI as input assuming that the data is unordered and irregular.

## Code:
## Training Model:
1. Prepare the horizontal and vertical EPIs using the method MultiEPL https://bit.ly/3Da8fB6.
2. Load dataset in teh format of numpy archives, and create grapphs using load_custom_data.py code.
3. To train the model, import functions from train_model.py file, and pass the parameters accordingly.
