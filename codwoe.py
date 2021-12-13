import json, sys
import numpy as np
import tensorflow as tf


class CodwoeTrainingSequence(tf.keras.utils.Sequence):
    '''Keras sequence for training the codwoe task.'''
    def __init__(self, embeddings, indexed_glosses, vocab_size, batch_size):
        self._embeddings = embeddings
        self._indexed_glosses = indexed_glosses
        self._max_gloss_length = max(len(g) for g in indexed_glosses)
        self._vocab_size = vocab_size
        self._batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self._embeddings) / self._batch_size))

    def __getitem__(self, batch_id):
        lower_bound = batch_id * self._batch_size
        upper_bound = min(lower_bound + self._batch_size, len(self._embeddings))

        x = np.zeros(
            (self._batch_size, self._max_gloss_length, len(self._embeddings[0])+self._vocab_size),
            dtype='float32',
        )
        y = np.zeros(
            (self._batch_size, self._max_gloss_length, self._vocab_size),
            dtype='float32',
        )

        for i, gloss in enumerate(self._indexed_glosses[lower_bound:upper_bound]):
            e = self._embeddings[i]
            for j, index in enumerate(gloss[:-1]):
                input_onehot = get_onehot(self._vocab_size, index)
                x[i][j] = np.concatenate((e, list(input_onehot)))

                output_onehot = get_onehot(self._vocab_size, gloss[j+1])
                y[i][j] = output_onehot

        return x, y


def get_onehot(size, index):
    '''Create a one-hot vector.

    Return: Vector with length `size` and `index` set to 1.

    '''
    onehot = np.zeros(size, dtype='float32')
    onehot[index] = 1.0
    return onehot


def preprocess_training_set(dataset, batch_size=128):
    '''Create a sequence for training, and accumulate a vocabulary.

    Return: sequence, vocabulary

    '''
    embeddings = []
    glosses = []
    max_gloss_length = 0
    v = set(['<g>', '</g'])
    print("Accumulating embeddings and vocabulary")
    for entry in dataset:
        # Build embeddings list
        embeddings.append(entry['sgns'])

        # Build glosses list
        glosses.append(
            ['<g>']
            + entry['gloss'].lower().split()
            + ['</g>']
        )

        # Update max_gloss_length
        if len(glosses[-1]) > max_gloss_length:
            max_gloss_length = len(glosses[-1])

        # Accumulate tokens to vocabulary
        for token in glosses[-1]:
            v.add(token)

    print("Indexing glosses")
    v = sorted(list(v))
    v_dict = {t: i for i, t in enumerate(v)} # faster lookup for token indices
    indexed_glosses = []
    for gloss in glosses:
        indexed_glosses.append([v_dict[t] for t in gloss])

    s = CodwoeTrainingSequence(embeddings, indexed_glosses, len(v), batch_size)

    return s, v


training_data_path = sys.argv[1] # TODO use multiple data sources
with open(training_data_path) as training_data_file:
    training_data = json.load(training_data_file)

sequence, vocabulary = preprocess_training_set(training_data, 128)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.LSTM(256, input_shape=(len(x_train[0]), len(x_train[0][0]))))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.Dense(1, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')

# model = keras.Model(inputs=inputs, outputs=outputs)
# model.compile()
# model.summary()

# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
