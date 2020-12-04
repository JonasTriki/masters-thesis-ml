from typing import List, Tuple

import tensorflow as tf

from word_embeddings.tokenizer import Tokenizer

AUTOTUNE = tf.data.experimental.AUTOTUNE


def generate_skip_gram_pairs(
    word_indices: tf.Tensor,
    max_window_size: int,
) -> tf.Tensor:
    """
    Generates skip-gram target/context pairs.

    Parameters
    ----------
    word_indices : tf.Tensor
        Tokenized words in a Tensor.
    max_window_size : int
        Maximum number of words to the left and right of the target word to generate positive samples from.
    """

    def skip_gram_pairs_from_word(
        word_index: int, skip_grams_array: tf.TensorArray
    ) -> Tuple[int, tf.TensorArray]:
        """
        Helper method for generating skip-gram target/context pairs from a single word integer.

        Parameters
        ----------
        word_index : int
            Word integer representation of word.
        skip_grams_array : tf.TensorArray
            TensorArray containing generated skip-gram target/context pairs.

        Returns
        -------
        next_word_index : int
            Next word_index to generate from.
        next_skip_grams_array : tf.TensorArray
            TensorArray containing newly generated skip-gram target/context pairs.
        """

        # Get word integer
        word_int = word_indices[word_index]

        # Randomly sample window size
        window_size = tf.random.uniform(
            [], minval=1, maxval=max_window_size + 1, dtype=tf.int32
        )

        # Generate positive samples
        left = tf.range(tf.maximum(word_index - window_size, 0), word_index)
        right = tf.range(
            word_index + 1,
            tf.minimum(word_index + 1 + window_size, tf.size(word_indices)),
        )
        context_indices = tf.concat([left, right], axis=0)
        context_word_indices = tf.gather(word_indices, context_indices)
        positive_samples = tf.stack(
            [tf.fill(tf.shape(context_word_indices), word_int), context_word_indices],
            axis=1,
        )

        return word_index + 1, skip_grams_array.write(word_index, positive_samples)

    size = tf.size(word_indices)
    # initialize a tensor array of length `tf.size(word_indices)`
    init_array = tf.TensorArray(tf.int64, size=size, infer_shape=False)
    _, result_array = tf.while_loop(
        lambda i, ta: i < size, skip_gram_pairs_from_word, [0, init_array]
    )
    instances = tf.cast(result_array.concat(), tf.int64)
    instances.set_shape([None, 2])

    return instances


def tokenize_and_subsample_words(
    text: tf.Tensor, word_keep_probs: tf.Tensor, tokenizer: Tokenizer
) -> tf.Tensor:
    """
    Tokenizes and applies subsampling to a text with a certain probability for each word.

    Parameters
    ----------
    text : tf.Tensor
        Text containing words separated by space.
    word_keep_probs : tf.Tensor
        Tensor containing probabilities of keeping a word during subsampling.

        It is ordered such that the first probability corresponds to the word with the
        most occurrences, the second probability to the second most occurring word, etc.
    tokenizer : Tokenizer
        Tokenizer instance to use for tokenizing texts.

    Returns
    -------
    word_indices_subsampled : tf.Tensor
        Tensor containing subsampled word indices.
    """

    # Tokenize text
    tokenized_text = tokenizer.tokenize_text_tf(text)

    # Filter out unknown words
    word_indices_filtered = tf.boolean_mask(
        tokenized_text, tf.not_equal(tokenized_text, tokenizer.unknown_word_int)
    )
    word_keep_probs_filtered = tf.gather(word_keep_probs, word_indices_filtered)

    # Generate random values from 0 to 1 to determine
    # if we keep or discard a word, with probabilities
    # defined in `word_keep_probs_filtered`
    rng_values = tf.random.uniform(
        tf.shape(word_keep_probs_filtered), 0, 1, dtype=tf.float64
    )

    # Subsample word indices
    word_indices_subsampled = tf.boolean_mask(
        word_indices_filtered, tf.less(rng_values, word_keep_probs_filtered)
    )

    return word_indices_subsampled


# Create dataset
def create_dataset(
    text_data_filepaths: List[str],
    num_texts: int,
    tokenizer: Tokenizer,
    max_window_size: int,
    batch_size: int,
) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset for training a word2vec model using skip-grams and negative sampling.

    Parameters
    ----------
    text_data_filepaths : list
        Paths of text data to generate skip-gram target/context pairs from.
    num_texts : int
        Number of texts (or sentences) in the text data file.
    tokenizer : Tokenizer
        Tokenizer instance for tokenizing individual texts.
    max_window_size : int
        Maximum number of words to the left and right of a target word during sampling of positive words.
    batch_size : int
        Number of skip-gram target/context pairs to yield for each batch of data.

    Returns
    -------
    dataset : tf.data.Dataset
        Dataset used for yielding skip-gram target/context pairs.
    """

    # Convert word keep probs to tensors
    word_keep_probs_tf = tf.convert_to_tensor(tokenizer.word_keep_probs)

    # Initialize tf.data.Dataset
    dataset = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(text_data_filepaths, num_parallel_reads=AUTOTUNE),
            tf.data.Dataset.from_tensor_slices(tf.range(num_texts) / num_texts),
        )
    )

    # Apply subsampling
    dataset = dataset.map(
        lambda text, sent_percentage: (
            tokenize_and_subsample_words(text, word_keep_probs_tf, tokenizer),
            sent_percentage,
        ),
        num_parallel_calls=AUTOTUNE,
    )

    # Filter out texts with less than 2 words in them
    dataset = dataset.filter(
        lambda word_indices, sent_percentage: tf.greater(tf.size(word_indices), 1),
    )

    # Generate skip-gram target/context pairs
    dataset = dataset.map(
        lambda word_indices, sent_percentage: (
            generate_skip_gram_pairs(
                word_indices,
                max_window_size,
            ),
            sent_percentage,
        ),
    )

    # Reshape `sent_percentage` to have the same size as `word_indices`
    dataset = dataset.map(
        lambda word_indices, sent_percentage: (
            word_indices,
            tf.fill(tf.shape(word_indices)[:1], sent_percentage),
        ),
        num_parallel_calls=AUTOTUNE,
    )

    # Create a dataset by unstacking word_indices
    dataset = dataset.flat_map(
        lambda word_indices, sent_percentages: tf.data.Dataset.from_tensor_slices(
            (word_indices, sent_percentages)
        ),
    )

    # Perform batching
    dataset = dataset.batch(batch_size, drop_remainder=True)

    def prepare_input_for_training(
        skip_gram_pairs_batch: tf.Tensor, sent_percentages: tf.Tensor
    ):
        """
        Prepares the target/context skip-gram pairs for training. Also converts
        the list of percentages into a single percentage.

        Parameters
        ----------
        skip_gram_pairs_batch : tf.Tensor
            Tensor containing input target/context pairs
        sent_percentages : tf.Tensor
            Tensor containing percentages of training progress
        Returns
        -------
        training_data : tuple of tf.Tensor
            Tuple consisting of targets, contexts and training progress (percentage).
        """

        # Set shape of tf.Tensor and extract targets/contexts
        skip_gram_pairs_batch.set_shape([batch_size, 2])
        input_targets = skip_gram_pairs_batch[:, :1]
        input_contexts = skip_gram_pairs_batch[:, 1:]

        # Ensure that dimensions are correct
        input_targets = tf.squeeze(input_targets, axis=1)
        input_contexts = tf.squeeze(input_contexts, axis=1)

        # Return percentage as a single number
        sent_percentage = sent_percentages[0]
        sent_percentage = tf.cast(sent_percentage, "float32")

        # Combine into tuple
        training_data = (input_targets, input_contexts, sent_percentage)

        return training_data

    # Prepare input for training
    dataset = dataset.map(
        lambda skip_gram_pairs_batch, sent_percentages: prepare_input_for_training(
            skip_gram_pairs_batch, sent_percentages
        ),
        num_parallel_calls=AUTOTUNE,
    )

    # Enable prefetching to prepare data while training
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
