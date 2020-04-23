from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Reshape, dot, Embedding, Dot

# Create the model
def build_word2vec_model_embedding(vocab_size: int, vector_dim: int):
    '''
    TODO: Docs
    '''
    # Input to network
    input_target = Input((1,), name='input_target')
    input_context = Input((1,), name='input_context')

    # Embedding layer
    embedding = Embedding(vocab_size + 1, vector_dim, input_length=1, name='embedding')
    target = embedding(input_target)
    target = Reshape((vector_dim, 1), name='target_word_vector')(target)
    context = embedding(input_context)
    context = Reshape((vector_dim, 1), name='context_word_vector')(context)
    
    # Compute (unnormalized) cosine similarity
    dot_product = Dot(axes=1, name='dot_product')([target, context])
    dot_product = Reshape((1,), name='dot_product_reshape')(dot_product)
    
    # Sigmoid activation (output)
    output = Dense(1, activation='sigmoid', name='sigmoid_activation')(dot_product)
    
    # Create model
    model = Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Create (slower) model using Dense instead of Embedding layer
def build_word2vec_model_dense(vocab_size: int, vector_dim: int):
    '''
    TODO: Docs
    '''
    # Input to network
    input_shape = (vocab_size + 1,)
    input_target = Input(input_shape, name='input_target')
    input_context = Input(input_shape, name='input_context')

    # Embedding layer using Dense layers
    embedding = Dense(vector_dim, input_shape=input_shape, name='embedding')
    target = embedding(input_target)
    target = Reshape((vector_dim, 1), name='target_word_vector')(target)
    context = embedding(input_context)
    context = Reshape((vector_dim, 1), name='context_word_vector')(context)
    
    # Compute (unnormalized) cosine similarity
    dot_product = Dot(axes=1, name='dot_product')([target, context])
    dot_product = Reshape((1,), name='dot_product_reshape')(dot_product)
    
    # Sigmoid activation (output)
    output = Dense(1, activation='sigmoid', name='sigmoid_activation')(dot_product)
    
    # Create model
    model = Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model