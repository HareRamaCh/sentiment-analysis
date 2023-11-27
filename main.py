import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment(text):
    # Tokenize and pad the input text
    text_sequence = imdb.get_word_index().get(text.lower(), 0)
    text_sequence = pad_sequences([[text_sequence]], maxlen=max_length)  # Wrap the integer in a list

    # Make predictions
    prediction = model.predict(text_sequence)

    if prediction >= 0.5:
        return ["Positive", prediction]
    else:
        return ["Negative", prediction]

vocab_size = 10000
max_length = 250
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length))
model.add(SimpleRNN(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 64
epochs = 3

# Store the training history to access loss and accuracy values per epoch
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(x_test, y_test))

# Plot training and validation loss over epochs
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Review Input
sample_review = input("Enter a sample review: ")
sentiment = predict_sentiment(sample_review)
print(f"Sentiment: {sentiment[0]}, Confidence: {sentiment[1]}")
