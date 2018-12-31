from keras.datasets import imdb

vocab_size = 8000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size, index_from=0)

word_to_id = imdb.get_word_index()
id_to_word = {value: key for key, value in word_to_id.items()}
print(' '.join(id_to_word[i] for i in X_train[0]))
