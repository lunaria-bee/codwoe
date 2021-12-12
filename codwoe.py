import json, sys

# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout, Masking, Embedding


# def embedding_to_gloss(e):
#     '''TODO'''
#     pass


data_path = sys.argv[1]
with open(data_path) as data_file:
    data = json.load(data_file)

embedding_matrix = [d['sgns'] for d in data]
