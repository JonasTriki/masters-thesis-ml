from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dot, Embedding, Input, Reshape
from tensorflow.keras.optimizers import Adam


def build_word2vec_model(
    vocab_size: int,
    vector_dim: int,
    learning_rate: float,
    target_embedding_layer_name: str = "target_embedding",
    model_name: str = "word2vec_sgns",
) -> Model:
    """
    Builds a Word2vec model using skip-gram negative sampling.

    Based on this paper by Mikolov et al.:
    https://arxiv.org/pdf/1301.3781v3.pdf

    Parameters
    ----------
    vocab_size : int
        Number of unique words we have in our vocabulary.
    vector_dim : int
        Latent embedding dimension to use for the target/embedding layers.
    learning_rate : float
        Learning rate.
    target_embedding_layer_name : str, optional
        Name to use for the target embedding layer (defaults to "target_embedding").
    model_name : str, optional
        Name of the model (defaults to "word2vec_sgns").

    Returns
    -------
    model : tf.keras.Model
        Keras Word2vec model
    """
    # Input to network
    input_shape = (1,)
    input_target = Input(input_shape, name="input_target")
    input_context = Input(input_shape, name="input_context")

    # Embedding layers
    target_embedding = Embedding(
        vocab_size, vector_dim, input_length=1, name=target_embedding_layer_name
    )
    target = target_embedding(input_target)
    target = Reshape((vector_dim,), name="target_word_vector")(target)
    context_embedding = Embedding(
        vocab_size, vector_dim, input_length=1, name="context_embedding"
    )
    context = context_embedding(input_context)
    context = Reshape((vector_dim,), name="context_word_vector")(context)

    # Compute (unnormalized) cosine similarity
    dot_product = Dot(axes=1, name="dot_product")([target, context])

    # Sigmoid activation (output)
    output = Dense(1, activation="sigmoid", name="sigmoid_activation")(dot_product)

    # Create model
    model = Model(inputs=[input_target, input_context], outputs=output, name=model_name)
    adam = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=adam)

    return model
