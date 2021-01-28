import tensorflow as tf


class NLP:
    def __init__(self, vocab_size, embedding_dim, input_length=None,
                 binary=True, weights=None):
        """
        :param vocab_size:
        :param embedding_dim
        :param input_length: if None, (None, )
        :param binary: if is a binary classification
        :param weights: if it has pretrained Embedding weights
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        if binary:
            self.final_activation = 'sigmoid'
        else:
            self.final_activation = 'softmax'

        embed_kwargs = {
            'input_dim': self.vocab_size,
            'output_dim': self.embedding_dim,
            'input_length': self.input_length
        }
        if weights is not None:
            embed_kwargs['weights'] = weights
            embed_kwargs['trainable'] = False
        self.embed_kwargs = embed_kwargs

    def vanilla_gap(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(**self.embed_kwargs),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation=self.final_activation)
        ])

    def bidirect_lstm(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(**self.embed_kwargs),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation=self.final_activation)
        ])

    def conv1d(self):
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(**self.embed_kwargs),
            tf.keras.layers.Conv1D(128, 5, activation='relu'),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation=self.final_activation)
        ])

    def avoid_overfitting(self):
        """ Answer for week3 exercise """
        assert self.embed_kwargs['weights'] is not None
        assert isinstance(self.embed_kwargs['weights'], list)
        return tf.keras.Sequential([
            tf.keras.layers.Embedding(**self.embed_kwargs),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(64, 5, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1, activation=self.final_activation)
        ])