#!/usr/bin/env python3

import argparse, json, sys
import numpy as np
import tensorflow as tf

from datetime import datetime


def create_parser_from_subcommand(subcommand):
    parser = argparse.ArgumentParser()

    if subcommand == 'train':
        parser.add_argument('training_data_path')
        parser.add_argument('embedding_type')
        parser.add_argument('-e', '--epochs')
        parser.add_argument('-c', '--checkpoint-path')

    elif subcommand == 'test':
        parser.add_argument('dev_data_path')
        parser.add_argument('model_path')
        parser.add_argument('embedding_type')

    else:
        print(
            f"Error: Subcommand must be one of 'test'|'train', not {subcommand}",
            file=sys.stderr,
        )

    parser.add_argument('-b', '--batch-size')

    return parser


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
            (self._batch_size, self._max_gloss_length-1, len(self._embeddings[0])+self._vocab_size),
            dtype='float32',
        )
        y = np.zeros(
            (self._batch_size, self._max_gloss_length-1, self._vocab_size),
            dtype='float32',
        )

        for i, gloss in enumerate(self._indexed_glosses[lower_bound:upper_bound]):
            e = self._embeddings[i]
            for j, index in enumerate(gloss[:-1]): # last element of gloss is not in input
                # Concatenate embedding and current step one-hot for x elt
                input_onehot = create_onehot(self._vocab_size, index)
                x[i][j] = np.concatenate((e, list(input_onehot)))

                # Use next step one-hot for y elt
                output_onehot = create_onehot(self._vocab_size, gloss[j+1])
                y[i][j] = output_onehot

        return x, y


def create_onehot(size, index):
    '''Create a one-hot vector.

    Return: Vector with length `size` and `index` set to 1.

    '''
    onehot = np.zeros(size, dtype='float32')
    onehot[index] = 1.0
    return onehot


def preprocess_training_set(dataset, embedding_type='sgns', batch_size=128):
    '''Create a sequence for training, and accumulate a vocabulary.

    Return: sequence, vocabulary

    '''
    embeddings = []
    glosses = []
    v = set(['<g>', '</g'])
    print("Accumulating embeddings and vocabulary")
    for entry in dataset:
        # Build embeddings list
        embeddings.append(entry[embedding_type])

        # Build glosses list
        glosses.append(
            ['<g>']
            + entry['gloss'].lower().split()
            + ['</g>']
        )

        # Accumulate tokens to vocabulary
        for token in glosses[-1]:
            v.add(token)

    print("Indexing glosses")
    v = sorted(list(v))
    v_dict = {t: i for i, t in enumerate(v)} # faster lookup for type indices
    indexed_glosses = []
    for gloss in glosses:
        indexed_glosses.append([v_dict[t] for t in gloss])

    s = CodwoeTrainingSequence(embeddings, indexed_glosses, len(v), batch_size)

    return s, v


def main(argv):
    if argv[1] == 'train':
        parser = create_parser_from_subcommand('train')
        args = parser.parse_args(argv[2:])
        training_data_path = args.training_data_path
        embedding_type = args.embedding_type
        batch_size = args.batch_size or 64
        epochs = args.epochs or 4
        checkpoint_path = args.checkpoint_path

        if args.embedding_type not in ('sgns', 'char', 'electra'):
            print(f"Error: embedding type must be one of 'sgns'|'char'|'electra', not"
                  + f" {embedding_type}")

        lang = training_data_path.split('/')[-1].split('.')[0] # TODO use Path for portability

        with open(training_data_path) as training_data_file:
            training_data = json.load(training_data_file)

        print(f"Training on {embedding_type}")
        sequence, vocabulary = preprocess_training_set(
            training_data,
            embedding_type=embedding_type,
            batch_size=batch_size,
        )

        if checkpoint_path:
            # Load from checkpoint if available
            print("Loading model")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            # Otherwise build a new model
            print("Building model")
            x_sample, y_sample = sequence[0]
            print(f"x dim: ({len(sequence)}, {x_sample.shape[1]}, {x_sample.shape[2]})")
            print(f"y dim: ({len(sequence)}, {y_sample.shape[1]}, {y_sample.shape[2]})")

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.LSTM(32, input_shape=(x_sample.shape[1], x_sample.shape[2])))
            model.add(tf.keras.layers.RepeatVector(y_sample.shape[1]))
            model.add(tf.keras.layers.Dense(len(vocabulary), activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')

            del x_sample, y_sample # free memory

        model.summary()

        print("Fitting model")
        timestr = datetime.today().strftime("%y%m%d_%H%M%S")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f'checkpoints/{lang}-{embedding_type}-checkpoint-{{loss:.4f}}-{timestr}.hdf5',
                monitor='loss',
                mode='min',
                save_best_only=True,
                verbose=1,
            ),
        ]
        model.fit(sequence, epochs=4, callbacks=callbacks)
        model.save(f'{lang}-{embedding_type}-model-{timestr}.h5')

    elif argv[1] == 'test':
        pass # TODO


if __name__ == '__main__':
    main(sys.argv)
