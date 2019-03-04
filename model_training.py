import pandas as pd
import numpy as np
import os
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras import Sequential
from keras.layers import Embedding, Dropout, SimpleRNN, Dense, LSTM
from collections import Counter


# def remove_pattern(input_txt, pattern):
#     r = re.findall(pattern, input_txt)
#     for i in r:
#         input_txt = re.sub(i, '', input_txt)
#     return input_txt


def read_file(filename):

    nSamples = 500000
    df = pd.read_csv(filename, encoding="ISO-8859-1", usecols=[0, 5])
    # 799647 pos and 248928 neg
    df = df.iloc[800000-nSamples//2: 800000+nSamples//2]
    df = df.sample(frac=1)
    df.iloc[:, 0] = df.iloc[:, 0].replace(4, 1)
    print(df.iloc[:, 0].value_counts())

    labels = list(df.iloc[:, 0])
    comments = list(df.iloc[:, 1])
    #comments = list(map(remove_pattern, comments, ["@[\w]*"]*len(comments)))

    for i in range(len(comments)):
        temp = re.sub('@[\w]*', '', comments[i])
        temp = re.sub('[^a-zA-Z]+', ' ', temp)
        comments[i] = ' '.join([w for w in temp.split() if len(w) > 3])

    return comments, labels

def prepare_data(comments, labels):

    ndata = len(comments)
    train_percent = 0.7
    vocabulary_size = 20000
    max_words = 100

    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(comments)
    print("number of words:", len(tokenizer.index_word))
    comments_seq = tokenizer.texts_to_sequences(comments)
    comments_seq_pad = sequence.pad_sequences(comments_seq, maxlen=max_words)

    train_x = comments_seq_pad[:int(ndata * train_percent)]
    test_x = comments_seq_pad[int(ndata*train_percent):]
    train_y = labels[:int(ndata * train_percent)]
    test_y = labels[int(ndata*train_percent):]

    print("train:", Counter(train_y))
    print("test:", Counter(test_y))

    return train_x, test_x, train_y, test_y, vocabulary_size, max_words

def network(train_x, test_x, train_y, test_y, vocabulary_size, max_words):

    embedding_size = 32
    batch_size = 64
    nepoch = 3

    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    #model.add(SimpleRNN(100))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())


    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=nepoch,
              validation_split=0.1,
              verbose=1)

    scores = model.evaluate(test_x, test_y, verbose=1)
    print('Accuracy:', scores[1])


if __name__ == "__main__":
    filename = os.getcwd() + "/twitter_1600k.csv"
    comments, labels = read_file(filename)
    train_x, test_x, train_y, test_y, vocabulary_size, max_words = prepare_data(comments, labels)
    network(train_x, test_x, train_y, test_y, vocabulary_size, max_words)
