import numpy as np
rng_seed = 399
np.random.seed(rng_seed)
import tensorflow as tf
tf.random.set_seed(rng_seed)
from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table
from tensorflow.keras.preprocessing.text import Tokenizer
AUTOTUNE = tf.data.experimental.AUTOTUNE

class SGNSDataLoader():
    '''
    Data loader for skipgram negative sampling target/context pairs and labels
    '''

    def __init__(
        self,
        texts: list,
        tokenizer: Tokenizer,
        batch_size: int,
        n_epochs: int,
        sampling_window_size: int,
        num_negative_samples: int,
        sampling_factor: float = 1e-5,
        one_hot_skipgram_pairs: bool = False,
        ):
        '''
        TODO: Docs
        '''
        self.texts = texts
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.vocab_size = len(tokenizer.word_index)
        self.sampling_table = make_sampling_table(self.vocab_size, sampling_factor=sampling_factor)
        self.sampling_window_size = sampling_window_size
        self.num_negative_samples = num_negative_samples

        self.one_hot_skipgram_pairs = one_hot_skipgram_pairs

    def _data_gen(self):
        '''
        TODO: Docs
        '''
        # Shuffle text in a random manner
        np.random.shuffle(self.texts)

        for text in self.texts:
            
            # Convert to sequence
            [sequence] = self.tokenizer.texts_to_sequences([text])

            # Create skipgram pairs
            skipgram_pairs, skipgram_labels = skipgrams(
                sequence,
                self.vocab_size,
                sampling_table=self.sampling_table,
                window_size=self.sampling_window_size,
                negative_samples=self.num_negative_samples
            )
            skipgram_pairs = np.array(skipgram_pairs)
            skipgram_labels = np.array(skipgram_labels)
            skipgram_data = np.column_stack((skipgram_pairs, skipgram_labels))

            # Yield one target-context pair at a time
            # e.g. (105, 200) --> 0
            for i in range(len(skipgram_data)):
                yield skipgram_data[i]

    def _skipgram_data_to_dict(self, skipgram_data_ragged_tensor: tf.RaggedTensor):
        '''
        TODO: Docs
        '''
        # Extract pairs and labels
        skipgram_data_tensor = skipgram_data_ragged_tensor.to_tensor()
        skipgram_pairs_target = skipgram_data_tensor[:, 0]
        skipgram_pairs_context = skipgram_data_tensor[:, 1]
        skipgram_labels = skipgram_data_tensor[:, 2]

        # One-hot encode target/context pairs
        if self.one_hot_skipgram_pairs:
            skipgram_pairs_target = tf.one_hot(skipgram_pairs_target, self.vocab_size + 1)
            skipgram_pairs_context = tf.one_hot(skipgram_pairs_context, self.vocab_size + 1)

        return (
            {
                'input_target': skipgram_pairs_target,
                'input_context': skipgram_pairs_context
            },
            skipgram_labels
        )

    def __call__(self):
        '''
        TODO: Docs
        '''
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_generator(
            self._data_gen,
            output_types=(tf.int64),
            output_shapes=(tf.TensorShape(3))
        )
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=self.batch_size))
        dataset = dataset.map(
            self._skipgram_data_to_dict,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.repeat(self.n_epochs)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        return dataset