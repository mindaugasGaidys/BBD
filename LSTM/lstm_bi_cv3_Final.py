import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, KFold
import tensorflow
import json
from keras.layers import LeakyReLU
from keras.layers import Bidirectional
from sklearn.metrics import confusion_matrix, classification_report

train_data = pd.read_excel('Data/Datasets/LSTM.xlsx', sheet_name='LSTM_train')
test_data = pd.read_excel('Data/Datasets/LSTM.xlsx', sheet_name='LSTM_test')

train_texts = train_data['text_cleaned'].values
train_categories = train_data['category'].values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
word_index = tokenizer.word_index

max_length = max([len(seq) for seq in train_sequences])
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding='post')

train_categories = to_categorical(train_categories, num_classes=3)

test_texts = test_data['text_cleaned'].values
test_categories = test_data['category'].values

test_sequences = tokenizer.texts_to_sequences(test_texts)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

test_categories = to_categorical(test_categories, num_classes=3)

vocab_size = len(word_index) + 1
embedding_dim = 300

def load_glove_embeddings(glove_file, word_index, embedding_dim):
    embeddings_index = {}
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError:
                continue
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    return embeddings_matrix

embeddings_file = 'C:\\Users\\hunde\\Desktop\\Bakalauras\\Scrap\\LSTM\\embeddings_matrix.npy'

if os.path.exists(embeddings_file):
    print("Embedding from file")
    embeddings_matrix = np.load(embeddings_file)
else:
    print("GloVe embedding")
    glove_file = 'Data/glove.840B.300d.txt'
    embeddings_matrix = load_glove_embeddings(glove_file, word_index, embedding_dim)
    np.save(embeddings_file, embeddings_matrix)

def create_model(lstm_layers=2, lstm_units=16, dropout_rate=0.5, dense_layers=1, dense_units=32, learning_rate=0.0001):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=[embeddings_matrix], input_length=max_length, trainable=True))

    for i in range(lstm_layers):
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=(i < lstm_layers - 1))))

    model.add(Dropout(dropout_rate))

    for _ in range(dense_layers):
        model.add(Dense(dense_units))
        model.add(LeakyReLU(alpha=0.001))
        model.add(Dropout(dropout_rate))

    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

data = padded_train_sequences
labels = train_categories
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

fold = 0
for train_index, val_index in kfold.split(data, labels):
    fold += 1
    
    X_train, X_val = data[train_index], data[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    model = create_model()
    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')

    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stop])

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model accuracy - Fold {fold}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f'LSTM/FinalResults/Fold_{fold}_LSTM_ACC.png')
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model loss - Fold {fold}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(f'LSTM/FinalResults/Fold_{fold}_LSTM_LOSS.png')
    plt.close()

_, accuracy = model.evaluate(padded_test_sequences, test_categories, verbose=0)

np.save('LSTM/FinalResults/trained_embedding.npy', model.layers[0].get_weights()[0])

y_pred = model.predict(padded_test_sequences)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(test_categories, axis=1)

cm = confusion_matrix(y_test_classes, y_pred_classes)
classification_metrics = classification_report(y_test_classes, y_pred_classes)

with open('LSTM/FinalResults/metrics.txt', 'w') as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClassification Metrics:\n")
    f.write(classification_metrics)

model_info = {
    'lstm_layers': 2,
    'lstm_units': 16,
    'dropout_rate': 0.5,
    'dense_layers': 1,
    'dense_units': 16,
    'learning_rate': 0.0001,
    'patience': 20,
    'batch_size': 64,
    'epochs': len(history.history['accuracy']),
    'model' : "Bidirectional",
    'test_accuracy': accuracy
}

with open('LSTM/FinalResults/model_info.txt', 'a') as outfile:
    json.dump(model_info, outfile)
    outfile.write('\n')

