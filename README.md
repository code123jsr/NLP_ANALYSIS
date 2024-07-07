# Natural Language Processing with Convolutional Neural Networks

This project aims to perform sentiment analysis on movie reviews using various Convolutional Neural Network (CNN) architectures. The dataset used is the Movie Review Dataset, which consists of positive and negative reviews. Different CNN architectures are experimented with to determine the most effective model for this task.

# File Structure
    - `NLP_Analysis.py`: This file contains a comprehensive script for performing sentiment analysis on the Movie Review Dataset using various Convolutional Neural Network (CNN) architectures. The script starts by downloading and extracting the dataset, followed by 
                         loading and preprocessing the data. It tokenizes the text reviews and creates a vocabulary based on word frequency. The reviews are then converted into sequences of token IDs and split into training and testing sets. The script also includes 
                         downloading and loading pre-trained Word2Vec embeddings from the Google News dataset to create an embedding matrix. Four different CNN architectures are defined and experimented with, each varying in the number of convolutional layers, fully 
                         connected layers, activation functions, and optimizers. The script trains these models on the training data, evaluates them on the testing data, and prints the training loss for each epoch and the final test accuracy. Hyperparameters such as 
                         learning rate, weight decay, and the number of epochs are tuned to find the optimal configuration.

# Dataset:

The dataset used is the rt-polaritydata from Cornell University, which consists of positive and negative movie reviews. It can be downloaded using the provided script:

```
wget http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz
tar -xvzf rt-polaritydata.tar.gz
```
# Usage
To run the code, use the following command
```
python NLP_Analysis.py
```

# Requirements:

- Python 3.x
- PyTorch
- TorchText
- spaCy
- Gensim
- scikit-learn
- requests
- portalocker
- numpy
- gzip
- shutil



