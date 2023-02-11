import pandas as pd
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data from the csv file into a DataFrame
df = pd.read_csv("dataset.csv")

# Check the shape of the DataFrame
print(df.shape)

# Print the first few rows of the DataFrame
print(df.head())


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
print(num_unique_words)



train_sentences = df.utterance
train_labels = df.intent

# Use label encoder to encode labels for the model
label_encoder = LabelEncoder()
label_encoder.fit(train_labels)
train_labels = label_encoder.transform(train_labels)

# Tokenize training data for the model
embedding_dim = 128
max_length = 100

tokenizer = Tokenizer(num_words = num_unique_words)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_length)
num_classes = 27



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


# Chat function
while True:

    print("You:")
    text = input()

    if text == 'quit':
        break

    input_data = pad_sequences(tokenizer.texts_to_sequences([text]), truncating='post', maxlen=max_length)

    prediction = model.predict(input_data, verbose=0)


    tag = label_encoder.inverse_transform([np.argmax(prediction)])

    print("Chatbot prediction: " + tag)

    print()