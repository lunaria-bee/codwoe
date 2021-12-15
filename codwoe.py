#!/usr/bin/env python3

import argparse, json, sys
import numpy as np
import tensorflow as tf

from datetime import datetime


def preprocess_training_set(dataset, embedding_type='sgns', vocabulary=None):
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
    if vocabulary is not None:
        v = vocabulary
    else:
        v = sorted(list(v))
    v_dict = {t: i for i, t in enumerate(v)} # faster lookup for type indices
    x = np.zeros((len(dataset), max_gloss_len-1, len(embeddings[0])+1))
    y = np.zeros((len(dataset), max_gloss_len-1, 1))
    for i, gloss in enumerate(glosses):
        e = embeddings[i]
        for j, token in enumerate(gloss[:-1]): # no input for final token
            if token in v_dict:
                x[i][j] = np.concatenate((e, [v_dict[token]]))
            else:
                x[i][j] = np.concatenate((e, [-1])) # use -1 for OOV tokens
            if gloss[j+1] in v_dict:
                y[i][j] = np.array([v_dict[gloss[j+1]]])
            else:
                y[i][j] = np.array([-1]) # use -1 for OOV tokens

    return x, y, v


def create_parser_from_subcommand(subcommand):
    parser = argparse.ArgumentParser()

    if subcommand == 'train':
        parser.add_argument('training_data_path')
        parser.add_argument('embedding_type', choices=('char', 'electra', 'sgns'))
        parser.add_argument('-e', '--epochs', type=int)
        parser.add_argument('-l', '--load')
        parser.add_argument('-o', '--output')
        parser.add_argument('-c', '--checkpoint-output')

    elif subcommand == 'test':
        parser.add_argument('model_path')
        parser.add_argument('dev_data_path')
        parser.add_argument('training_data_path')
        parser.add_argument('embedding_type', choices=('char', 'electra', 'sgns'))

    else:
        print(f"Error: Subcommand must be one of 'test'|'train', not {subcommand}", file=sys.stderr)

    parser.add_argument('-b', '--batch_size', type=int)

    return parser


def main(argv):
    if argv[1] == 'train':
        parser = create_parser_from_subcommand('train')
        args = parser.parse_args(argv[2:])
        training_data_path = args.training_data_path
        embedding_type = args.embedding_type
        batch_size = args.batch_size or 64
        epochs = args.epochs or 20
        load_path = args.load
        output_path = args.output or './'
        checkpoint_path = args.checkpoint_output or 'checkpoints/'

        lang = training_data_path.split('/')[-1].split('.')[0] # TODO use Path for portability

        with open(training_data_path) as training_data_file:
            training_data = json.load(training_data_file)

        print(f"Training on {embedding_type}")
        x_train, y_train, vocabulary = preprocess_training_set(training_data, embedding_type)

        if load_path:
            # Load model/checkpoint from file if available
            print("Loading model")
            model = tf.keras.models.load_model(load_path)
        else:
            # Otherwise build a new model
            print("Building model")

            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.LSTM(32, input_shape=(x_train.shape[1], x_train.shape[2])))
            model.add(tf.keras.layers.RepeatVector(y_train.shape[1]))
            model.add(tf.keras.layers.Dense(len(vocabulary), activation='softmax'))
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        print(f"x dim: {x_train.shape}")
        print(f"y dim: {y_train.shape}")
        model.summary()

        print("Fitting model")
        timestr = datetime.today().strftime("%y%m%d_%H%M%S")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f'{checkpoint_path}/{lang}-{embedding_type}-checkpoint-{{loss:.4f}}-{timestr}.hdf5',
                monitor='loss',
                mode='min',
                save_best_only=True,
                verbose=1,
            ),
        ]
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        model.save(f'{output_path}/{lang}-{embedding_type}-model-{timestr}.h5')

    elif argv[1] == 'test':
        parser = create_parser_from_subcommand('test')
        args = parser.parse_args(argv[2:])
        model_path = args.model_path
        dev_data_path = args.dev_data_path
        training_data_path = args.training_data_path
        embedding_type = args.embedding_type
        batch_size = args.batch_size

        # Load files
        print("Loading data")
        model = tf.keras.models.load_model(model_path)
        model.compile()
        with open(training_data_path) as training_data_file:
            training_data = json.load(training_data_file)
        with open(dev_data_path) as dev_data_file:
            dev_data = json.load(dev_data_file)

        x_train, y_train, vocabulary = preprocess_training_set(training_data, embedding_type)
        x_dev, y_target, _ = preprocess_training_set(dev_data, embedding_type, vocabulary)

        initial_token = vocabulary.index('<g>')
        final_token = vocabulary.index('</g>')

        # Initialize input
        print("Initializing input")
        input_ = np.zeros((x_dev.shape[0], x_train.shape[1], x_dev.shape[2]))
        for i in range(input_.shape[0]):
            input_[i][0] = x_dev[i][0]

        # Initialize prediction
        print("Initializing prediction")
        y_pred = np.zeros((y_target.shape[0], y_train.shape[1], y_target.shape[2]))
        for i in range(y_pred.shape[0]):
            y_pred[i][0] = np.array([initial_token])

        # Predict
        print("Predicting")
        finished = set()

        # For each potential gloss element (which is the second dimension of each matrix,
        # hence iterating on `j` at outer level)
        for j in range(x_train.shape[1]-1):
            print(f"gloss token {j+1}/{x_train.shape[1]-1}")

            # Predict in batches to preserve memory
            batch_count = x_dev.shape[0]//batch_size
            for batch in range(batch_count):
                lower_bound = batch*batch_size
                upper_bound = min(lower_bound+batch_size, x_dev.shape[0])
                print(f"  batch {batch+1}/{batch_count}, {lower_bound}:{upper_bound}")
                prediction = model.predict_on_batch(input_[lower_bound:upper_bound])

                # For each example (which is the first dimension of each matrix, hence
                # iterating on `i` at the inner level)
                for i in range(lower_bound, upper_bound):
                    # Assign predictions for unfinished glosses
                    if i not in finished:
                        e = x_dev[i][0][:-1]
                        y_pred[i][j+1] = np.array([np.argmax(prediction[i-lower_bound][j+1])])
                        input_[i][j+1] = np.concatenate((e, y_pred[i][j]))

                        # Once we reach </g>, mark gloss completed so we stop generating for it
                        if y_pred[i][j+1][0] == final_token:
                            finished.add(i)

        # If any gloss reached the end without predicting </g>, force set last element to </g>
        for i in range(y_pred.shape[0]):
            if i not in finished:
                y_pred[i][-1] = np.array([final_token])

        # Write results
        output = []
        for i in range(y_pred.shape[0]):
           output.append({
               'id': dev_data[i]['id'],
               'gloss': ' '.join([vocabulary[int(y[0])] for y in y_pred[i][1:-1]]),
           })

        output_path = model_path.replace('.h5', '-results.json')
        with open(output_path, 'w') as f:
            json.dump(output, f)

        cross_entropy = numpy.mean(
            tf.keras.metrics.sparse_categorical_crossentropy(
                y_target,
                y_pred,
                from_logits=True,
            )
        )
        print(f"Cross-entropy: {cross_entropy}")

    else:
        print(f"Error: Subcommand must be one of 'train'|'test', not {argv[1]}", file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv)
