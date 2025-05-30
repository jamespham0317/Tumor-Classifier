# Tumor Classification using CNNs

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify histopathology images of tumors as either benign or malignant using the 
[BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis).

## Model Summary

- **Architecture**: Basic CNN with 2 convolutional blocks and dense layers
- **Input Size**: 64x64 RGB images
- **Loss Function**: Binary Crossentropy
- **Activation Function**: ReLU for hidden layers, sigmoid for binary output 
- **Optimizer**: Adam
- **Performance**: ~87% training accuracy, ~81% validation accuracy
- **EarlyStopping** used to avoid overfitting
