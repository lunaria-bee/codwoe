#!/usr/bin/env python3

import argparse, json, sys
import numpy as np
import tensorflow as tf

from datetime import datetime


def preprocess_training_set(dataset, embedding_type='sgns'):
    '''Create a sequence for training, and accumulate a vocabulary.

    Return: sequence, vocabulary

    '''
    embeddings = []
    glosses = []
    v = set(['<g>', '</g'])
    max_gloss_len = 0
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

        # Update max gloss length
        if len(glosses[-1]) > max_gloss_len:
            max_gloss_len = len(glosses[-1])

        # Accumulate tokens to vocabulary
        for token in glosses[-1]:
            v.add(token)

    print("Arranging inputs and outputs")
    v = sorted(list(v))
    v_dict = {t: i for i, t in enumerate(v)} # faster lookup for type indices
    x = np.zeros((len(dataset), max_gloss_len-1, len(embeddings[0])+1))
    y = np.zeros((len(dataset), max_gloss_len-1))
    for i, gloss in enumerate(glosses):
        e = embeddings[i]
        for j, token in enumerate(gloss[:-1]): # no input for final token
            x[i][j] = np.concatenate((e, [v_dict[token]]))
            y[i][j] = v_dict[gloss[j+1]]

    return x, y, v


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

    return parser


def main(argv):
    if argv[1] == 'train':
        parser = create_parser_from_subcommand('train')
        args = parser.parse_args(argv[2:])
        training_data_path = args.training_data_path
        embedding_type = args.embedding_type
        epochs = args.epochs or 4
        checkpoint_path = args.checkpoint_path

        if args.embedding_type not in ('sgns', 'char', 'electra'):
            print(f"Error: embedding type must be one of 'sgns'|'char'|'electra', not"
                  + f" {embedding_type}")

        lang = training_data_path.split('/')[-1].split('.')[0] # TODO use Path for portability

        with open(training_data_path) as training_data_file:
            training_data = json.load(training_data_file)

        print(f"Training on {embedding_type}")
        x_train, y_train, vocabulary = preprocess_training_set(training_data, embedding_type)

        if checkpoint_path:
            # Load from checkpoint if available
            print("Loading model")
            model = tf.keras.models.load_model(checkpoint_path)
        else:
            # Otherwise build a new model
            print("Building model")
            print(f"x dim: {x_train.shape}")
            print(f"y dim: {y_train.shape}")

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])))
            model.add(tf.keras.layers.Dense(len(vocabulary), activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

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
        model.fit(sequence, epochs=epochs, callbacks=callbacks)
        model.save(f'{lang}-{embedding_type}-model-{timestr}.h5')

    elif argv[1] == 'test':
        pass # TODO


if __name__ == '__main__':
    main(sys.argv)
