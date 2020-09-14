"""Defines word2vec model using tf.keras API.
"""
import tensorflow as tf


class Word2VecSGNSModel(tf.keras.Model):
    """Word2Vec model."""

    def __init__(
        self,
        unigram_counts,
        hidden_size=300,
        batch_size=256,
        negatives=5,
        power=0.75,
        alpha=0.025,
        min_alpha=0.0001,
        add_bias=True,
        random_seed=0,
    ):
        """Constructor.

        Args:
          unigram_counts: a list of ints, the counts of word tokens in the corpus.
          hidden_size: int scalar, length of word vector.
          batch_size: int scalar, batch size.
          negatives: int scalar, num of negative words to sample.
          power: float scalar, distortion for negative sampling.
          alpha: float scalar, initial learning rate.
          min_alpha: float scalar, final learning rate.
          add_bias: bool scalar, whether to add bias term to dot product
            between target- and context embedding vectors.
          random_seed: int scalar, random_seed.
        """
        super(Word2VecSGNSModel, self).__init__()
        self._unigram_counts = unigram_counts
        self._hidden_size = hidden_size
        self._vocab_size = len(unigram_counts)
        self._batch_size = batch_size
        self._negatives = negatives
        self._power = power
        self._alpha = alpha
        self._min_alpha = min_alpha
        self._add_bias = add_bias
        self._random_seed = random_seed

        self.add_weight(
            "target_embedding",
            shape=[self._vocab_size, self._hidden_size],
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.5 / self._hidden_size, maxval=0.5 / self._hidden_size
            ),
        )

        self.add_weight(
            "context_embedding",
            shape=[self._vocab_size, self._hidden_size],
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
        )

        self.add_weight(
            "biases", shape=[self._vocab_size], initializer=tf.keras.initializers.Zeros()
        )

    def call(self, inputs, labels):
        """Runs the forward pass to compute loss.
        Uses negative sampling to compute loss.

        Args:
          inputs: int tensor of shape [batch_size]
          labels: int tensor of shape [batch_size]

        Returns:
          loss: float tensor, cross entropy loss, of shape [batch_size, negatives + 1].
        """
        target_embedding, context_embedding, biases = self.weights

        sampled_values = tf.random.fixed_unigram_candidate_sampler(
            true_classes=tf.expand_dims(labels, 1),
            num_true=1,
            num_sampled=self._batch_size * self._negatives,
            unique=True,
            range_max=len(self._unigram_counts),
            distortion=self._power,
            unigrams=self._unigram_counts,
        )

        sampled = sampled_values.sampled_candidates
        sampled_mat = tf.reshape(sampled, [self._batch_size, self._negatives])
        inputs_target_embedding = tf.gather(
            target_embedding, inputs
        )  # [batch_size, hidden_size]
        true_context_embedding = tf.gather(
            context_embedding, labels
        )  # [batch_size, hidden_size]
        # [batch_size, negatives, hidden_size]
        sampled_context_embedding = tf.gather(context_embedding, sampled_mat)
        # [batch_size]
        true_logits = tf.reduce_sum(
            tf.multiply(inputs_target_embedding, true_context_embedding), axis=1
        )
        # [batch_size, negatives]
        sampled_logits = tf.einsum(
            "ijk,ikl->il",
            tf.expand_dims(inputs_target_embedding, 1),
            tf.transpose(sampled_context_embedding, (0, 2, 1)),
        )

        if self._add_bias:
            # [batch_size]
            true_logits += tf.gather(biases, labels)
            # [batch_size, negatives]
            sampled_logits += tf.gather(biases, sampled_mat)

        # [batch_size]
        true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits
        )
        # [batch_size, negatives]
        sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits
        )

        loss = tf.concat(
            [tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], axis=1
        )
        return loss
