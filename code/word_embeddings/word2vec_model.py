from typing import List

import tensorflow as tf


class Word2VecSGNSModel(tf.keras.Model):
    """
    Word2Vec skip-gram negative sampling Keras model.
    """

    def __init__(
        self,
        word_counts: List[int],
        embedding_dim: int = 300,
        batch_size: int = 256,
        num_negative_samples: int = 5,
        unigram_exponent_negative_sampling: float = 0.75,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        add_bias: bool = True,
        name: str = "word2vec_sgns",
        target_embedding_layer_name: str = "target_embedding",
        **kwargs
    ):
        """Initializes the word2vec skip-gram negative sampling Keras model

        Parameters
        ----------
        word_counts : a list of ints
            The counts of word tokens in the corpus.
        embedding_dim : int scalar
            Length of word vector.
        batch_size : int scalar
            Batch size.
        num_negative_samples : int scalar
            Number of negative words to sample.
        unigram_exponent_negative_sampling : float scalar
            Distortion for negative sampling.
        learning_rate : float scalar
            Initial learning rate.
        min_learning_rate : float scalar
            Final learning rate.
        add_bias : bool scalar
            Whether to add bias term to dot product between target and context embedding vectors.
        name : str
            Name of the model
        target_embedding_layer_name : str
            Name to use for the target embedding layer (defaults to "target_embedding").
        """
        super(Word2VecSGNSModel, self).__init__(name=name, **kwargs)
        self._word_counts = word_counts
        self._embedding_dim = embedding_dim
        self._vocab_size = len(word_counts)
        self._batch_size = batch_size
        self._num_negative_samples = num_negative_samples
        self._unigram_exponent_negative_sampling = unigram_exponent_negative_sampling
        self._learning_rate = learning_rate
        self._min_learning_rate = min_learning_rate
        self._add_bias = add_bias
        self._target_embedding_layer_name = target_embedding_layer_name

        self.add_weight(
            self._target_embedding_layer_name,
            shape=[self._vocab_size, self._embedding_dim],
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.5 / self._embedding_dim, maxval=0.5 / self._embedding_dim
            ),
        )

        self.add_weight(
            "context_embedding",
            shape=[self._vocab_size, self._embedding_dim],
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
        )

        self.add_weight(
            "biases", shape=[self._vocab_size], initializer=tf.keras.initializers.Zeros()
        )

    def get_config(self) -> dict:
        """
        Gets the config for the word2vec model.
        """
        config = {
            "word_counts": self._word_counts,
            "embedding_dim": self._embedding_dim,
            "vocab_size": self._vocab_size,
            "batch_size": self._batch_size,
            "num_negative_samples": self._num_negative_samples,
            "unigram_exponent_negative_sampling": self._unigram_exponent_negative_sampling,
            "learning_rate": self._learning_rate,
            "min_learning_rate": self._min_learning_rate,
            "add_bias": self._add_bias,
        }
        return config

    def call(self, input_targets: tf.Tensor, input_contexts: tf.Tensor) -> tf.Tensor:
        """
        Runs the forward pass to compute loss. Uses negative sampling to compute loss.

        Parameters
        ----------
        input_targets: int tensor of shape [batch_size]
            Input targets to train on.
        input_contexts: int tensor of shape [batch_size]
            Input contexts to train on.

        Returns
        -------
        loss: float tensor
            Cross entropy loss, of shape [batch_size, negatives + 1].
        """
        target_embedding, context_embedding, biases = self.weights

        # Positive samples
        # [batch_size, hidden_size]
        inputs_target_embedding = tf.gather(target_embedding, input_targets)
        # [batch_size, hidden_size]
        inputs_context_embedding = tf.gather(context_embedding, input_contexts)

        # Multiply target and context embeddings to get (unnormalized) cosine similarities
        # [batch_size]
        positive_logits = tf.reduce_sum(
            tf.multiply(inputs_target_embedding, inputs_context_embedding), axis=1
        )

        # Negative samples
        negative_sampler = tf.random.fixed_unigram_candidate_sampler(
            true_classes=tf.expand_dims(
                tf.range(self._batch_size, dtype=tf.int64), axis=0
            ),
            num_true=self._batch_size,
            num_sampled=self._batch_size * self._num_negative_samples,
            unique=True,
            range_max=len(self._word_counts),
            distortion=self._unigram_exponent_negative_sampling,
            unigrams=self._word_counts,
        )
        negative_samples = negative_sampler.sampled_candidates
        negative_samples_mat = tf.reshape(
            negative_samples, [self._batch_size, self._num_negative_samples]
        )
        # [batch_size, negatives, hidden_size]
        negative_samples_embedding = tf.gather(context_embedding, negative_samples_mat)

        # Multiply target embeddings with embeddings of negative samples to get
        # (unnormalized) cosine similarities
        # [batch_size, negatives]
        negative_logits = tf.einsum(
            "ijk,ikl->il",
            tf.expand_dims(inputs_target_embedding, 1),
            tf.transpose(negative_samples_embedding, (0, 2, 1)),
        )

        # Add bias
        if self._add_bias:
            # [batch_size]
            positive_logits += tf.gather(biases, input_contexts)
            # [batch_size, negatives]
            negative_logits += tf.gather(biases, negative_samples_mat)

        # Use cross-entropy to compute losses for both positive and negative logits
        # [batch_size]
        positive_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(positive_logits), logits=positive_logits
        )
        # [batch_size, negatives]
        negative_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(negative_logits), logits=negative_logits
        )

        # Merge losses together into a single loss
        loss = tf.concat(
            [tf.expand_dims(positive_cross_entropy, 1), negative_cross_entropy], axis=1
        )
        return loss
