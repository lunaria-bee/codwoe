import math
import numpy as np
import tensorflow as tf


GLOSS_START = '<g>'
GLOSS_END = '</g>'
GLOSS_EMPTY = '<EMPTY>'
GLOSS_UNK = '<UNK>'


class CodwoeTrainingSequence(tf.keras.utils.Sequence):
    '''Sequence for batching training input.

    Initializers
    embeddings: Input embedddings.
    glosses: Output glosses.
    vocabulary: Indexed vocabulary of types.
    batch_size: Size of batches to generate.
    max_gloss_len: Optional. Maximum length to allow for glosses, including gloss start
                   and gloss end symbols ('<g>' and '</g>'). If this argument is omitted,
                   maximum gloss length will be calculated from the longest gloss in
                   `glosses`. Note that any glosses that exceed this length will be
                   truncated to fit.

    '''
    def __init__(self, embeddings, glosses, vocabulary, batch_size, max_gloss_len=None):
        self._embeddings = embeddings
        self._glosses = glosses
        self._vocabulary = vocabulary
        self._batch_size = batch_size

        # We need to make sure max_gloss_len is set, even though it can also be calculated
        # in `make_xy_matrices()`, because if we let the calculation fall through to
        # `make_xy_matrices()` it will only calculate max_gloss_len for the glosses in the
        # current batch.
        if max_gloss_len is None:
            self._max_gloss_len = max(len(g) for g in glosses)
        else:
            self._max_gloss_len = max_gloss_len

    def __len__(self):
        return math.ceil(len(self._embeddings) / self._batch_size)

    def __getitem__(self, batch_index):
        lower_bound = batch_index * self._batch_size
        upper_bound = min(lower_bound+self._batch_size, len(self._embeddings)-1)
        return make_xy_matrices(
            self._embeddings[lower_bound:upper_bound],
            self._glosses[lower_bound:upper_bound],
            self._vocabulary,
            self._max_gloss_len,
        )

    @property
    def x_shape(self):
        return (len(self._embeddings), self._max_gloss_len-1, len(self._embeddings[0])+1)

    @property
    def y_shape(self):
        return (len(self._glosses), self._max_gloss_len-1, 1)


def preprocess_labeled_data(
        dataset,
        embedding_type='sgns',
        vocabulary=None,
        verbose=True,
):
    '''Preprocess dataset that has both inputs and labels.

    Parameters
    dataset: Data to preprocess.
    embedding_type: Which embedding to use for input. Default 'sgns'.
    vocabulary: Optional. Previously accumulated vocabulary to use instead of accumulating
                a new one. Must meet the following criteria:
                - must be an indexable container (i.e. implements `__getitem__()`),
                  ideally a list or tuple
                - must contain at least '<g>', '</g>', '<EMPTY>', and '<UNK>'
                - '<EMPTY>' must appear at the 0th index
                - must contain no repeated elements
                Failure to meet these criteria will result in a ValueError; a vocabulary
                returned from this function will always meet these criteria. Any tokens
                that appear in `dataset`'s glosses that are not in `vocabulary` will be
                marked '<UNK>'.
    verbose: Whether to print output regarding progress. Default `True`.

    Return: embeddings, indexed_glosses, vocabulary
    embeddings: Input embeddings from dataset.
    indexed_glosses: Output glosses from dataset, with each token represented as an index
                     into `vocabulary`.
    vocabulary: Indexed vocabulary of types. Identical to `vocabulary` argument if
                supplied, accumulated from `dataset` if not.

    '''
    # If no vocabulary is provided via argument, accumulate a new one
    accumulate_vocabulary = (vocabulary is None)
    if accumulate_vocabulary:
        # Initialize vocabulary for accumulation
        v = set([GLOSS_START, GLOSS_END, GLOSS_UNK])
    else:
        # Check that `vocabulary` is valid
        if not hasattr(vocabulary, '__getitem__'):
            raise ValueError(f"`vocabulary` must be an indexable type, not {type(vocabulary)}")

        if any(t not in vocabulary for t in (GLOSS_START, GLOSS_END, GLOSS_UNK, GLOSS_EMPTY)):
            raise ValueError(
                f"`vocabulary` must contain '{GLOSS_START}', '{GLOSS_END}', '{GLOSS_UNK}',"
                + f" and '{GLOSS_EMPTY}'"
            )

        if vocabulary[0] != GLOSS_EMPTY:
            raise ValueError(
                f"'{GLOSS_EMPTY}' must be at index 0 of `vocabulary`, not index"
                + f" {vocabulary.index(GLOSS_EMPTY)}"
            )

        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("`vocabulary` must not contain any repeated elements")

    embeddings = []
    raw_glosses = []

    if verbose:
        print("Accumulating embeddings", end="")
        if accumulate_vocabulary:
            print(" and vocabulary")
        else:
            print()
    for entry in dataset:
        # Build embeddings list
        embeddings.append(entry[embedding_type])

        # Build raw_glosses list
        new_gloss = [GLOSS_START] + entry['gloss'].lower().split() + [GLOSS_END]
        raw_glosses.append(new_gloss)

        # Accumulate tokens to vocabulary
        if accumulate_vocabulary:
            v = v.union(set(new_gloss))

    if accumulate_vocabulary:
        v = ['<EMPTY>'] + sorted(list(v))
    else:
        v = vocabulary
    v_dict = {t: i for i, t in enumerate(v)} # faster lookup for type indices

    indexed_glosses = []
    for i, raw_gloss in enumerate(raw_glosses):
        indexed_glosses.append([v_dict.setdefault(t, GLOSS_UNK) for t in raw_gloss])

    return embeddings, indexed_glosses, v


def make_xy_matrices(embeddings, glosses, vocabulary, max_gloss_len=None):
    '''Construct training input/output matrices.

    Parameters
    embeddings: Input embeddings.
    glosses: Output glosses, represented as indices of `vocabulary`.
    vocabulary: Indexed vocabulary of types. Should be ordered such that indices in
                `glosses` correspond to a type in `vocabulary`.
    max_gloss_len: Optional. Maximum length to allow for glosses, including gloss start
                   and gloss end symbols ('<g>' and '</g>'). If this argument is omitted,
                   maximum gloss length will be calculated from the longest gloss in
                   `glosses`. Note that any glosses that exceed this length will be
                   truncated to fit.

    Return: x, y
    x: Model input. Concatenation of input embedding and vocabulary type index for each
       token in each gloss.
    y: Model output. Expected next type index for each step of input.

    '''
    if max_gloss_len is None:
        max_gloss_len = max(len(g) for g in glosses)

    x = np.zeros((len(embeddings), max_gloss_len-1, len(embeddings[0])+1))
    y = np.zeros((len(glosses), max_gloss_len-1, 1))

    # Needed to indicate end of gloss when truncating to max_gloss_len
    gloss_end_index = vocabulary.index(GLOSS_END)

    for i in range(x.shape[0]):
        e = embeddings[i]
        g = glosses[i]
        # Iterate through gloss, cutting iteration off at max_gloss_len if necessary
        for j in range(min(len(g)-2, x.shape[1])):
            t = g[j]
            t_next = g[j+1]
            x[i][j] = np.concatenate((e, [t]))
            y[i][j] = np.array([t_next])
        # If gloss was truncated to max_gloss_len, force last token to GLOSS_END
        if len(g) > max_gloss_len:
            y[i][-1] = np.array([gloss_end_index])

    return x, y
