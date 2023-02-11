"""
Training the model.
"""


import pandas as pd
import numpy as np
import pickle
from collections import Counter

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


# Load the data from the csv file into a DataFrame
df = pd.read_csv("dataset.csv")

# Check the shape of the DataFrame
print(df.shape)

# Print the first few rows of the DataFrame
# print(df.head())

# Using the counter, we iterate over our data column and count all the unique words
def counter_word(desc_col):
  count = Counter()
  for desc in desc_col.values:
    for word in desc.split():
      count[word] += 1
  return count

counter = counter_word(df.utterance)

# The length of the counter is the number of unique words, which we store in a variable for later
num_unique_words = len(counter)

# Our data has 27 unique labels
num_classes = 27

# Get our training sentences and labels from the corresponding columns in the dataset
train_sentences = df.utterance
train_labels = df.intent

# Use label encoder to encode labels for the model
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)
train_labels = label_encoder.transform(train_labels)

# Initial embedding size 128 and max length 100 as a standard
embedding_dim = 128
max_length = 100

# Tokenize training data for the model
tokenizer = Tokenizer(num_words = num_unique_words)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_length)

"""
Creating the model. Embedding layer with initial standard size of 128 neurons,
GlobalAveragePooling1D to capture sentence meaning and solve a classification problem (since our chatbot has to classify 27 different intents)
Two more dense layers with 64 and total intents number of neurons with softmax activation function for non-binary classification.
"""
model = Sequential()
model.add(Embedding(num_unique_words, embedding_dim, input_length=max_length))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

# Number of epochs set to 10
epochs = 10
history = model.fit(padded_sequences, np.array(train_labels), epochs=epochs)

# Saving trained model
model.save("chatbot_trained_model")

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the label encoder
with open('encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)