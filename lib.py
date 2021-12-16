import numpy as np


GLOSS_START = '<g>'
GLOSS_END = '</g>'
GLOSS_EMPTY = '<EMPTY>'
GLOSS_UNK = '<UNK>'


def preprocess_labeled_data(
        dataset,
        embedding_type='sgns',
        vocabulary=None,
        max_gloss_len=None,
        verbose=True,
):
    '''Preprocess dataset that has both inputs and labels.

    Parameters
    dataset: Data to preprocess.
    embedding_type: Which embedding to use for input. Default 'sgns'.
    vocabulary: Optional. Previously accumulated vocabulary to use instead of accumulating
                a new one. Must meet the following criteria:
                - must be iterable
                - must contain at least '<g>', '</g>', '<EMPTY>', and '<UNK>'
                - '<EMPTY>' must appear at the 0th index
                - must contain no repeated elements
                Failure to meet these criteria will result in a ValueError; a vocabulary
                returned from this function will always meet these criteria. Any tokens
                that appear in `dataset`'s glosses that are not in `vocabulary` will be
                marked '<UNK>'.
    max_gloss_len: Optional. Maximum length to allow for glosses, including gloss start
                   and gloss end symbols ('<g>' and '</g>'). If this argument is omitted,
                   maximum gloss length will be calculated from the glosses in
                   `dataset`. Note that any glosses in `dataset` that exceed this length
                   will be truncated to fit.
    verbose: Whether to print output regarding progress. Default `True`.

    Return: x, y, v
    x: Input for model.
    y: Expected output from model, based on labels.
    v: Vocabulary. Identical to `vocabulary` argument if supplied, accumulated from
       `dataset` if not.

    '''
    embeddings = []
    glosses = []

    # If no maximum gloss length was provided via argument, calculate a new one
    calculate_max_gloss_len = (max_gloss_len is None)
    if calculate_max_gloss_len:
        max_gloss_len = 0

    # If no vocabulary is provided via argument, accumulate a new one
    accumulate_vocabulary = (vocabulary is None)
    if accumulate_vocabulary:
        # Initialize vocabulary for accumulation
        v = set([GLOSS_START, GLOSS_END, GLOSS_UNK])
    else:
        # Check that `vocabulary` is valid
        if not hasattr(vocabulary, '__iter__'):
            raise ValueError(f"`vocabulary` must be an iterable type, not {type(vocabulary)}")

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

    if verbose:
        print("Accumulating embeddings", end="")
        if accumulate_vocabulary:
            print(" and vocabulary")
        else:
            print()
    for entry in dataset:
        # Build embeddings list
        embeddings.append(entry[embedding_type])

        # Build glosses list
        new_gloss = [GLOSS_START] + entry['gloss'].lower().split() + [GLOSS_END]
        if not calculate_max_gloss_len and len(new_gloss) > max_gloss_len:
            # If max_gloss_len was supplied by the caller and this gloss exceeds it,
            # truncate to fit
            new_gloss = new_gloss[:max_gloss_len]
            new_gloss[-1] = GLOSS_END
        glosses.append(new_gloss)

        # Update max gloss length
        if calculate_max_gloss_len and len(glosses[-1]) > max_gloss_len:
            max_gloss_len = len(glosses[-1])

        # Accumulate tokens to vocabulary
        if accumulate_vocabulary:
            for token in glosses[-1]:
                v.add(token)

    if accumulate_vocabulary:
        v = ['<EMPTY>'] + sorted(list(v))
    else:
        v = vocabulary
    v_dict = {t: i for i, t in enumerate(v)} # faster lookup for type indices

    if verbose:
        print("Arranging inputs and outputs")
    x = np.zeros((len(dataset), max_gloss_len-1, len(embeddings[0])+1))
    y = np.zeros((len(dataset), max_gloss_len-1, 1))
    for i, gloss in enumerate(glosses):
        e = embeddings[i]
        for j, token in enumerate(gloss[:-1]): # no input for final token
            if token in v_dict:
                x[i][j] = np.concatenate((e, [v_dict[token]]))
            else:
                x[i][j] = np.concatenate((e, [v_dict['<UNK>']]))
            if gloss[j+1] in v_dict:
                y[i][j] = np.array([v_dict[gloss[j+1]]])
            else:
                y[i][j] = np.array([v_dict['<UNK>']])

    return x, y, v
