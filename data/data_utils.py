import os
import pickle


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    with open(check_extension(filename), "wb") as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), "rb") as f:
        return pickle.load(f)
