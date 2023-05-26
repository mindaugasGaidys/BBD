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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import make_scorer, recall_score
import tensorflow as tf
import json
from keras.layers import LeakyReLU
from keras.layers import Bidirectional
from keras.metrics import Recall
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


# užkraunami duomenys
data_train = pd.read_excel('Data/Datasets/LSTM.xlsx', sheet_name='Sheet2')
texts_train = data_train['text_cleaned'].values
categories_train = data_train['category'].values
label_encoder = LabelEncoder()
y_train_numerical = label_encoder.fit_transform(categories_train)
#priskiriami svoriai
class_weights = compute_class_weight("balanced", classes=np.unique(categories_train), y=categories_train)
class_weight_dict = dict(enumerate(class_weights))
# ukraunami
data_test = pd.read_excel('Data/Datasets/LSTM.xlsx', sheet_name='LSTM_test')
texts_test = data_test['text_cleaned'].values
categories_test = data_test['category'].values

# Tokenize the texts and pad the sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts_train)
sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)
word_index = tokenizer.word_index
max_length = max([len(seq) for seq in sequences_train])
padded_sequences_train = pad_sequences(sequences_train, maxlen=max_length, padding='post')
padded_sequences_test = pad_sequences(sequences_test, maxlen=max_length, padding='post')

# Convert the categories to one-hot encoding
categories_train = to_categorical(categories_train, num_classes=3)
categories_test = to_categorical(categories_test, num_classes=3)

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
embeddings_file = 'LSTM/embeddings_matrix.npy'



if os.path.exists(embeddings_file):
    print("Embedding from file")
    embeddings_matrix = np.load(embeddings_file)
else:
    print("Loading GloVe embeddings...")
    glove_file = 'Data/glove.840B.300d.txt'
    embeddings_matrix = load_glove_embeddings(glove_file, word_index, embedding_dim)
    np.save(embeddings_file, embeddings_matrix)
    print("GloVe embeddings loaded and saved to file.")

def custom_recall_scorer(y_true, y_pred):
    y_pred_classes = np.argmax(y_pred, axis=-1)
    y_true_classes = np.argmax(y_true, axis=-1)
    _, recall, _, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, zero_division=0)
    recall_2 = recall[2]
    return recall_2

custom_scorer = make_scorer(custom_recall_scorer, greater_is_better=True)

def load_existing_model_info(filename):
    existing_models = []

    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, "r") as f:
            for line in f:
                existing_models.append(json.loads(line))
    return existing_models
model_info_file = 'model_info_recall_CW.txt'
existing_models = load_existing_model_info(model_info_file)
unique_params = {tuple(config.items()) for config in existing_models}


def create_model(lstm_layers=8, lstm_units=16, dropout_rate=0.3, dense_layers=3, dense_units=32, learning_rate=0.001, patience=20, batch_size=64):
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
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=[Recall()])

    model_info = {
        'lstm_layers': lstm_layers,
        'lstm_units': lstm_units,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'dense_units': dense_units,
        'dense_layers': dense_layers
    }
    print(f"Model parameters: {model_info}")
    if tuple(model_info.items()) not in unique_params:
        unique_params.add(tuple(model_info.items()))
        X_train, X_val, y_train, y_val = train_test_split(padded_sequences_train, categories_train, test_size=0.5, random_state=42, shuffle=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
        history = model.fit(X_train, y_train, epochs=55, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stop], class_weight=class_weight_dict)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        plt.figure()
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('Model recall')
        plt.ylabel('Recall')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(f'Data/Graphs/lstm_allData3catGloVeBidirectionalLeakyReLu_recall/LSTM_Recall_{timestamp}.png')
        plt.close()

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.savefig(f'Data/Graphs/lstm_allData3catGloVeBidirectionalLeakyReLu_recall/LSTM_LOSS_{timestamp}.png')
        plt.close()
        y_pred = model.predict(X_test)
        test_recall = custom_recall_scorer(y_test, y_pred)
        model_info['recall'] = test_recall
        model_info['timestamp'] = timestamp
        model_info['epochs'] = len(history.history['recall'])
        model_info['patience'] = early_stop.patience
    y_pred_classes = np.argmax(y_pred, axis=-1)
    y_test_classes = np.argmax(y_test, axis=-1)
    conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
    conf_matrix_str = str(conf_matrix).replace('\n', '\n\t')
    classification_metrics = classification_report(y_test_classes, y_pred_classes)

    with open('LSTM/model_info_recall.txt', 'a') as outfile:
        json.dump(model_info, outfile)
        outfile.write('Bidirectional, no SMOTE CW, small but balanced dataset, leakyRelu\n')
        outfile.write('Confusion Matrix:\n\t')
        outfile.write(conf_matrix_str)
        outfile.write("\n\nClassification Metrics:\n")
        outfile.write(classification_metrics)
        outfile.write('\n')
    return model
X_train, y_train = padded_sequences_train, categories_train
X_test, y_test = padded_sequences_test, categories_test

params = {
    'lstm_layers': [2,3],
    'lstm_units': [16,32],
    'dropout_rate': [0.1,0.3,0.5],
    'learning_rate': [0.0001,0.001],
    'dense_layers': [2,3],
    'dense_units': [16,32],
    'batch_size': [32,64],
    'patience': [15]
}
model = KerasClassifier(build_fn=create_model, lstm_layers=None, lstm_units=None, dropout_rate=None, dense_layers=None, dense_units=None, learning_rate=None, patience=20, verbose=1)
random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=4, scoring=custom_scorer, cv=2, n_jobs=-1, random_state=42)
random_search_result = random_search.fit(X_train, y_train)