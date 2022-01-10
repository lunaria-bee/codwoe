#!/usr/bin/env python3

from lib import *

import argparse, json, multiprocessing, sys
import numpy as np
import tensorflow as tf

from datetime import datetime
from pathlib import Path


SUBCOMMAND_CHOICES = ('pretrain', 'train', 'test')
EMBEDDING_TYPE_CHOICES = ('char', 'electra', 'sgns')
EPOCHS_DEFAULT = 20
OUTPUT_PATH_DEFAULT = './'
PRETRAINING_PATH_DEFAULT = 'pretrain/'
CHECKPOINT_PATH_DEFAULT = 'checkpoints/'
BATCH_SIZE_DEFAULT = 64
WORD2VEC_MAX_BATCH_SIZE = 10000


class SetJobLimitAction(argparse.Action):
    '''Action to limit number of threads (jobs) tensorflow can use.'''
    def __call__(self, parser, namespace, value, option_string=None):
        print("Setting job limit")
        tf.config.threading.set_intra_op_parallelism_threads(value)
        tf.config.threading.set_inter_op_parallelism_threads(value)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'subcommand',
        metavar='subcommand',
        choices=SUBCOMMAND_CHOICES,
        help=f"Must be one of {SUBCOMMAND_CHOICES}.",
    )
    args = parser.parse_args(argv[1:2]) # parse subcommand

    if args.subcommand == 'pretrain':
        parser.add_argument(
            'training_data_path',
            type=Path,
            help="Path to training data.",
        )
        parser.add_argument(
            '-p', '--pretraining-output',
            dest='pretraining_path',
            type=Path,
            default=PRETRAINING_PATH_DEFAULT,
            help=f"Path to pretraining output directory. Default='{PRETRAINING_PATH_DEFAULT}'.",
        )
        parser.add_argument(
            '--batch_pretraining',
            action='store_true',
            help=("Batch pretraining data. Pretraining is much less memory-intensive than"
                  + " training, and so is \"unbatched\" by default. (More precisely,"
                  + " without this argument, the maximum word2vec batch size of 10,000 is"
                  + " used regardless of the -b/--batch-size setting.)"),
        )

    elif args.subcommand == 'train':
        parser.add_argument(
            'training_data_path',
            type=Path,
            help="Path to training data.",
        )
        parser.add_argument(
            'embedding_type',
            metavar='embedding_type',
            choices=EMBEDDING_TYPE_CHOICES,
            help=f"Must be one of {EMBEDDING_TYPE_CHOICES}.",
        )
        parser.add_argument(
            '-e', '--epochs',
            type=int,
            default=EPOCHS_DEFAULT,
            help=f"Number of epochs over which to train model. Defaul={EPOCHS_DEFAULT}.",
        )
        parser.add_argument(
            '-l', '--load',
            type=Path,
            dest='load_path',
            help="Path to existing model or checkpoint from which to begin training.",
        )
        parser.add_argument(
            '-o', '--output',
            dest='output_path',
            type=Path,
            default=OUTPUT_PATH_DEFAULT,
            help=f"Path to model output directory. Default='{OUTPUT_PATH_DEFAULT}'.",
        )
        parser.add_argument(
            '-c', '--checkpoint-output',
            dest='checkpoint_path',
            type=Path,
            default=CHECKPOINT_PATH_DEFAULT,
            help=f"Path to checkpoint output directory. Default='{CHECKPOINT_PATH_DEFAULT}'.",
        )
        parser.add_argument(
            '-p', '--pretrain-output',
            dest='pretrain_path',
            type=Path,
            default=PRETRAINING_PATH_DEFAULT,
            help=f"Path to pretraining output directory. Default='{PRETRAIN_PATH_DEFAULT}'.",
        )
        parser.add_argument(
            '--batch-input',
            action='store_true',
            help=("Process input in batches. By default, only the model's output and"
                  + " hidden layers are batched."),
        )
        parser.add_argument(
            '--batch_pretraining',
            action='store_true',
            help=("Batch pretraining data. Pretraining is much less memory-intensive than"
                  + " training, and so is \"unbatched\" by default. (More precisely,"
                  + " without this argument, the maximum word2vec batch size of 10,000 is"
                  + " used regardless of the -b/--batch-size setting.)"),
        )

    elif args.subcommand == 'test':
        parser.add_argument(
            'dev_data_path',
            type=Path,
            help="Path to dev test data.",
        )
        parser.add_argument(
            'training_data_path',
            type=Path,
            help="Path to training data.",
        )
        parser.add_argument(
            'model_path',
            type=Path,
            help="Path to model.",
        )
        parser.add_argument(
            'embedding_type',
            choices=('char', 'electra', 'sgns'),
            help="Must be one of 'char'|electra'|'sgns'.",
        )

    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help=(f"Size of input batches to model. Set to 0 to disable batching (NOT"
              + f" RECOMMENDED). Default={BATCH_SIZE_DEFAULT}."),
    )
    parser.add_argument(
        '-j', '--jobs',
        action=SetJobLimitAction,
        type=int,
        help="Maximum number of jobs (i.e. threads) to run simultaneously.",
    )

    return parser.parse_args(argv[1:])


def pretrain(
        training_data_path='',
        pretraining_path=PRETRAINING_PATH_DEFAULT,
        batch_pretraining=False,
        batch_size=WORD2VEC_MAX_BATCH_SIZE,
        jobs=None,
):
    '''TODO'''
    if not batch_pretraining:
        batch_size = WORD2VEC_MAX_BATCH_SIZE

    if jobs is None:
        jobs = multiprocessing.cpu_count()

    lang = training_data_path.parts[-1].split('.')[0]

    with open(training_data_path) as training_data_file:
        training_data = json.load(training_data_file)

    print("Pretraining")
    pretraining_model, vocabulary = word2vec(training_data, batch_size, jobs)
    pretraining_path.mkdir(exist_ok=True)
    pretraining_model.save(f'{pretraining_path}/{lang}.embeddings.pickle')


def train(
        training_data_path='',
        embedding_type='',
        batch_size=BATCH_SIZE_DEFAULT,
        batch_input=False,
        batch_pretraining=False,
        epochs=EPOCHS_DEFAULT,
        load_path=None,
        output_path=OUTPUT_PATH_DEFAULT,
        checkpoint_path=CHECKPOINT_PATH_DEFAULT,
        pretraining_path=PRETRAINING_PATH_DEFAULT,
        jobs=None,
):
    '''Train a model.

    All parameters correspond to command-line arguments.

    '''
    lang = training_data_path.parts[-1].split('.')[0]

    with open(training_data_path) as training_data_file:
        training_data = json.load(training_data_file)

    # Automatically pretrain, if necessary
    embeddings_path = Path(f'{pretraining_path}/{lang}.embeddings.pickle')
    if not embeddings_path.is_file():
        pretrain(training_data_path, pretraining_path, batch_pretraining, batch_size, jobs)

    print(f"Training on {embedding_type}")
    embeddings, glosses, vocabulary = preprocess_labeled_data(training_data, embedding_type)
    if batch_input:
        sequence = CodwoeTrainingSequence(embeddings, glosses, vocabulary, batch_size)
        x_shape = sequence.x_shape
        y_shape = sequence.y_shape
    else:
        x_train, y_train = make_xy_matrices(embeddings, glosses, vocabulary)
        x_shape = x_train.shape
        y_shape = y_train.shape

    if load_path:
        # Load model/checkpoint from file if available
        print("Loading model")
        model = tf.keras.models.load_model(load_path)
    else:
        # Otherwise build a new model
        print("Building model")

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(32, input_shape=(x_shape[1], x_shape[2])))
        model.add(tf.keras.layers.RepeatVector(y_shape[1]))
        model.add(tf.keras.layers.Dense(len(vocabulary), activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    print(f"x dim: {x_shape}")
    print(f"y dim: {y_shape}")
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
    if batch_size == 0:
        batch_size = x_train.shape[0]

    if batch_input:
        print("Fitting with batched input")
        model.fit(sequence, epochs=epochs, callbacks=callbacks)
    else:
        print("Fitting with full input")
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    model.save(f'{output_path}/{lang}-{embedding_type}-model-{timestr}.h5')


def test(
        dev_data_path='',
        training_data_path='',
        model_path='',
        embedding_type='',
        batch_size=BATCH_SIZE_DEFAULT,
        jobs=None,
):
    '''Test a model.

    All parameters correspond to command-line arguments.

    '''
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

    if batch_size == 0:
        batch_size = x_dev.shape[0]

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


def main(argv):
    args = parse_args(argv)
    subcommand =  args.subcommand
    args_dict = vars(args)
    args_dict.pop('subcommand')

    if subcommand == 'pretrain':
        pretrain(**args_dict)

    elif subcommand == 'train':
        train(**args_dict)

    elif subcommand == 'test':
        test(**args_dict)


if __name__ == '__main__':
    main(sys.argv)
