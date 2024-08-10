# Sentiment Analysis Using Recurrent Neural Networks (RNN) with IMDB Dataset

## Objective

The objective of this assignment is to implement a Recurrent Neural Network (RNN) using TensorFlow to perform sentiment analysis on the IMDB movie review dataset. The goal is to train the RNN to classify movie reviews as positive or negative and analyze the model's performance.

## Project Steps

### Step 1: Dataset Preparation

- **Use the IMDB Dataset:**
  - Utilize the IMDB movie review dataset via Kaggle: https://www.kaggle.com/code/bilalsaeed06/movies-sentiment-analysis-lstm-rnn/input

- **Load and Preprocess the Data:**
  - **Tokenization:**
    - Tokenize the text and convert tokens to numerical format.
  - **Padding:**
    - Pad sequences to ensure uniform input length.

### Step 2: Building the RNN Model

- **Define the Model Architecture:**
  - Implement an RNN model using TensorFlow and Keras with the following layers:
    - **Input Layer:** Accepts the tokenized input data.
    - **Embedding Layer:** Converts token indices into dense vectors of fixed size.
    - **RNN Layer:** Use LSTM or GRU layers for better performance.
    - **Fully Connected Layer:** Dense layer to learn complex patterns.
    - **Output Layer:** Produces the final classification output (positive or negative).

- **Compile the Model:**
  - Compile the model with an appropriate loss function and optimizer.

### Step 3: Training the Model

- **Split the Dataset:**
  - Divide the dataset into training and validation sets.

- **Train the Model:**
  - Train the RNN model on the training set and validate it on the validation set.
  - Monitor the training process and employ techniques like early stopping if necessary.

### Step 4: Evaluating the Model

- **Performance Metrics:**
  - Evaluate the model’s performance using metrics such as accuracy and loss.

- **Visualization:**
  - Plot the training and validation loss and accuracy over epochs to visualize the training process.

- **Analysis:**
  - Analyze the results and discuss the performance of the RNN model.

### Step 5: Hyperparameter Tuning

- **Experiment with Hyperparameters:**
  - Adjust different hyperparameters (e.g., number of layers, units in each layer, dropout rate, learning rate) to enhance the model’s performance.

### Step 6: Comparative Analysis

- **Implement an Alternative Model:**
  - Implement another neural network architecture (e.g., a simple feedforward neural network) for the same sentiment analysis task.

- **Compare Models:**
  - Compare the performance of the RNN with the alternative model.


## Project Structure

The project directory contains the following files:

- **`.gitignore`**           : Specifies files and directories to be ignored by Git
- **`README.md`**            : Project documentation
- **`app.py`**               : Main application script
- **`best_model.h5`**        : Serialized best model for prediction
- **`movie_e\review.ipynb`** : Jupyter Notebook with data exploration and model training
- **`requirements.txt`**     : Python package dependencies
- **`tokenizer.pkl`**        : Convert text to sequences


## Installation

To set up the project, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/floramaevillarin/RNN_Movie_Review.git
