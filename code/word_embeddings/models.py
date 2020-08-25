from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dot, Embedding, Input, Reshape
from tensorflow.keras.optimizers import Adam


def build_word2vec_model(
    vocab_size: int, vector_dim: int, learning_rate: float = 0.001
) -> Model:
    """
    Builds a Word2vec model using skipgram negative sampling

    Based on this paper by Mikolov et al.:
    https://arxiv.org/pdf/1301.3781v3.pdf
    """
    # Input to network
    input_shape = (1,)
    input_target = Input(input_shape, name="input_target")
    input_context = Input(input_shape, name="input_context")

    # We add 1 to the vocabulary size to account for
    # the unknown word
    vocab_size_new = vocab_size + 1

    # Embedding layers
    target_embedding = Embedding(
        vocab_size_new, vector_dim, input_length=1, name="target_embedding"
    )
    target = target_embedding(input_target)
    target = Reshape((vector_dim,), name="target_word_vector")(target)
    context_embedding = Embedding(
        vocab_size_new, vector_dim, input_length=1, name="context_embedding"
    )
    context = context_embedding(input_context)
    context = Reshape((vector_dim,), name="context_word_vector")(context)

    # Compute (unnormalized) cosine similarity
    dot_product = Dot(axes=1, name="dot_product")([target, context])

    # Sigmoid activation (output)
    output = Dense(1, activation="sigmoid", name="sigmoid_activation")(dot_product)

    # Create model
    model = Model(inputs=[input_target, input_context], outputs=output)
    adam = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=adam)

    return model
