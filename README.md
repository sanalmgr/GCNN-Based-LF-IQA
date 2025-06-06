# GCNN-Based-LF-IQA

Paper: <a href="https://www.researchgate.net/publication/392020870_Assessing_the_Quality_of_Light_Field_Images_A_Graph-based_Approach">Assessing the Quality of Light Field Images: A Graph-based Approach</a>

In this work, we propose a no-reference LF-IQA method that predicts the quality of compressed LF images using a Deep Graph Convolutional Neural Network (GCNN-LFIQA). The GCNN-LFIQA method is based on a deep single-stream network architecture which takes horizontal EPI as input assuming that the data is unordered and irregular.

## Code:
## Training Model:
1. Prepare the horizontal and vertical EPIs using the method MultiEPL https://bit.ly/3Da8fB6.
2. Load the dataset in the format of numpy archives, and create graphs using load_custom_data.py code.
3. To train the model, import functions from train_model.py file, and pass the parameters accordingly.

Requirements:
- Python: >=3.6.0, <3.8.0
- Networkx: https://networkx.org/
- StellarGraph: https://stellargraph.readthedocs.io/en/stable/README.html
- Numpy
