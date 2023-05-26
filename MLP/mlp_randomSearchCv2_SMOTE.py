import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
import json
import os
from imblearn.over_sampling import SMOTE
from keras.layers import Flatten
from keras.layers import LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import make_scorer, precision_recall_fscore_support

def custom_recall_scorer(y_true, y_pred):
    y_pred_classes = np.argmax(y_pred, axis=-1)
    y_true_classes = np.argmax(y_true, axis=-1)
    _, recall, _, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, zero_division=0)
    recall_2 = recall[2]
    return recall_2

recall_scorer = make_scorer(custom_recall_scorer)

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

data = pd.read_excel('Data/Datasets/MLP.xlsx', sheet_name='bal')
test_data = pd.read_excel('Data/Datasets/MLP.xlsx', sheet_name='MLP_test')

texts = data['text_cleaned'].values
categories = data['category'].values
test_texts = test_data['text_cleaned'].values
test_categories = test_data['category'].values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')
categories = to_categorical(categories, num_classes=3)
test_categories = to_categorical(test_categories, num_classes=3)
vocab_size = len(word_index) + 1

glove_file_path = 'Data/glove.840B.300d.txt' 
embedding_dim = 300 

embeddings_file = 'MLP/embeddings_matrix.npy'

if os.path.exists(embeddings_file):
    print("Embedings from file")
    embedding_matrix = np.load(embeddings_file)
else:
    print("From GloVe")
    glove_file = 'Data/glove.840B.300d.txt'
    embedding_matrix = load_glove_embeddings(glove_file, word_index, embedding_dim)
    np.save(embeddings_file, embedding_matrix)
    print("GloVe embeddings loaded and saved to file.")

def create_model(hidden_layers=1, hidden_units=16, dropout_rate=0.5, learning_rate=0.001, patience=5, batch_size=64):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=True))
    model.add(Flatten())
    model.add(Dense(hidden_units, input_shape=(max_length,)))
    model.add(LeakyReLU())
    model.add(Dropout(dropout_rate))
    
    for _ in range(hidden_layers - 1):
        model.add(Dense(hidden_units))
        model.add(LeakyReLU())
        model.add(Dropout(dropout_rate))

    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=[tf.keras.metrics.Recall(name='recall')])
    
    model_info = {
        'hidden_layers': hidden_layers,
        'hidden_units': hidden_units,
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate
    }

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, categories, test_size=0.2, random_state=42, shuffle=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    history = model.fit(X_train_smote, y_train_smote, epochs=50, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure()
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig(f'Data/Graphs/mlp_allData3catCV2/MLP_RECALL_{timestamp}.png')
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig(f'Data/Graphs/mlp_allData3catCV2/MLP_LOSS_{timestamp}.png')
    plt.close()

    _, test_recall = model.evaluate(test_padded_sequences, test_categories, verbose=0)

    test_y_pred = model.predict(test_padded_sequences)
    test_y_pred_classes = np.argmax(test_y_pred, axis=-1)
    test_y_true_classes = np.argmax(test_categories, axis=-1)

    test_conf_matrix = confusion_matrix(test_y_true_classes, test_y_pred_classes)
    test_classification_metrics = classification_report(test_y_true_classes, test_y_pred_classes)
    model_info['recall'] = test_recall
    model_info['timestamp'] = timestamp
    model_info['epochs'] = len(history.history['recall'])
    model_info['patience'] = early_stop.patience
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_info_file = f'MLP/MLPresults/model_info_{timestamp}.txt'
    with open(model_info_file, 'a')as outfile:
        json.dump(model_info, outfile)
        outfile.write('\n')
        outfile.write('Confusion Matrix:\n\t')
        outfile.write(np.array2string(test_conf_matrix, separator=', '))
        outfile.write("\n\nClassification Metrics:\n")
        outfile.write(test_classification_metrics)
        outfile.write('\n')
        outfile.write('\n')

    return model

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, categories, test_size=0.2, random_state=42)

params = {
    'hidden_layers': [3],
    'hidden_units': [64,128],
    'dropout_rate': [0.5],
    'learning_rate': [0.0001],
    'batch_size': [128],
    'patience': [5]
}
model = KerasClassifier(build_fn=create_model, hidden_layers=None, hidden_units=None, dropout_rate=None, learning_rate=None, patience=5, verbose=1)
random_search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=10, scoring=recall_scorer, cv=2, n_jobs=-1, random_state=42)
random_search_result = random_search.fit(X_train, y_train)
