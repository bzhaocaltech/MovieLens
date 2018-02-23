# This file loads the data from training_data.txt and test_data.txt and the
# dumps them using pickle

import numpy as np
import pickle

movies = np.genfromtxt("data/movies.txt", dtype=None, delimiter = "\t", encoding = "ISO-8859-1")
pickle.dump(movies, open("data/movies.p", "wb"))

ratings = np.genfromtxt("data/data.txt", dtype = int, delimiter = "\t")
pickle.dump(ratings, open("data/ratings.p", "wb"))

Y_train = np.loadtxt('data/train.txt').astype(int)
pickle.dump(Y_train, open("data/y_train.p", "wb"))

Y_test = np.loadtxt('data/test.txt').astype(int)
pickle.dump(Y_test, open("data/y_test.p", "wb"))
