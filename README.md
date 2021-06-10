# END2_0_Session_5
Sentiment analysis of Stanford dataset


# Sentiment Analysis

![Sentiment Analysis](https://www.kdnuggets.com/wp-content/uploads/ambalina-sentiment-analysis-header.jpg)

Sentiment analysis is the process of processing a piece of text, typically in natural language and predicting the sentiment of it. This typically relies on using a previously annotated dataset. Using the dataset, an AI model can be trained to predict the sentiment of some new text.

In this assignment, the Stanford Sentiment Treebank dataset is used.


# Data Pre-Processing


## Transform the Labels

The input labels are converted from float to a range of 0-4. These labels (0-4) is what the model would see during training and learn to predict them.


## Data Augmentation

Before we can use the dataset for training our model, it needs to be augmented so that the data mimics the real world scenarios. The following data augmentations are performed on it -
* synonyms
* random character swap
* random word delete
* random spelling mistakes
* back translation (english to german and german to english)

Note: The above augmentation operations on the raw data are done using the nlpaug library.


## Training, Validation and Test Datasets 

After the data augmentation, the input data is split into training, vaidation and test dataasets (60%, 20%, 20% split) using numpy.


## Word Embeddings

In order to enhance the performance of the model, the Glove Vector representation of words is used. The advantage of GloVe is that, unlike Word2vec, GloVe does not rely just on local statistics (local context information of words), but incorporates global statistics (word co-occurrence) to obtain word vectors.


# Model


## Setup

The model is trained and run using the GPU available in Google Collab.


## Model Definition

The model is defined using Pytorch. It is a combination of multiple LSTM and Fully Connected layers. Multiple variation of the model have been trained and tested, more detials in the summary section below. 

The optimizer used is Adam. Advantages of Adam are -
* Straightforward to implement
* Computationally efficient 
* Little memory requirements 
* Invariant to diagonal rescale of the gradients

The loss function used is Cross Entropy. It is definitely a good loss function for classification problems, because it minimizes the distance between two probability distributions - predicted and actual.


## Model Training

The model is run for different combinations of epochs and input data. The model is evaluated on the basis of training and validation loss and accuracy.


# Predictions

Finally, the trained model is run against, "unseen data" (which is not present in the input training, validation and test datasets) to get an idea of the real world predictions.


# Summary

1 sample prediction is listed in table below. For all the other predictions, please refer the notebooks.

No  | Data Augmentation | Glove | Model Parameters | Training Results | Predictions | Jupyter Notebook |
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
1 | Synonym Augmentation | 6B, 100 Dimensions | Epoch 20, Layers 2 | Train Acc: 85.79%, Val. Acc: 51.69%, Test Acc: 52.85% | `predict_sentiment("Make no mistake, this predictable movie is clearly part of the Mission: Impossible franchise -- by which we mean it checks off all the usual boxes.")` very positive | [Synonym Augmentation Notebook](https://github.com/rajanm/END2_0_Session_5/blob/main/Stanford_Sentiment_Analysis_using_LSTM_RNN_Synonym_Augmentation.ipynb)|
2 | Synonym, Char Swap, Delete, Back Translation Augmentation | 6B, 300 Dimensions | Epoch 20, Layers 2 | Train Acc: 93.88%, Val. Acc: 64.23%, Test Acc: 65.53% | `predict_sentiment("Make no mistake, this predictable movie is clearly part of the Mission: Impossible franchise -- by which we mean it checks off all the usual boxes.")` positive | [Multiple Data Augmentations Notebook](https://github.com/rajanm/END2_0_Session_5/blob/main/Stanford_Sentiment_Analysis_using_LSTM_RNN_Multiple_Augmentation.ipynb) |
3 | Synonym, Char Swap, Delete, Back Translation Augmentation | 6B, 300 Dimensions | Epoch 40, Layers 2 | Train Acc: 93.74%, Val. Acc: 68.43%, Test Acc: 67.61%  | `predict_sentiment("Make no mistake, this predictable movie is clearly part of the Mission: Impossible franchise -- by which we mean it checks off all the usual boxes.")` negative | [Multiple Data Augmentations V2 Notebook](https://github.com/rajanm/END2_0_Session_5/blob/main/Stanford_Sentiment_Analysis_using_LSTM_RNN_Multiple_Augmentation_V2.ipynb) |
4 | Synonym, Char Swap, Delete, Spelling Mistake, Back Translation Augmentation | 6B, 300 Dimensions | Epoch 40, Layers 4 | Train Acc: 93.17%, Val. Acc: 68.43%, Test Acc: 67.97%  | `predict_sentiment("Make no mistake, this predictable movie is clearly part of the Mission: Impossible franchise -- by which we mean it checks off all the usual boxes.")` negative | [Multiple Data Augmentations V2 Notebook](https://github.com/rajanm/END2_0_Session_5/blob/main/Stanford_Sentiment_Analysis_using_LSTM_RNN_Multiple_Augmentation_V3.ipynb) |
