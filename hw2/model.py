import tensorflow as tf

class Model:
    def __init__(self):
        self.max_seq_length = 30
        self.word_embedding_size = 200
        self.vocab_size = 1193514

        self.rnn_size = 256

        self.layer_size = 2
        self.dropoutKeepProb = 0.8

        self.placeholders = self.__get_placeholders()
        self.build_gragh()

    def __get_placeholders(self):
        self.X_ph = tf.placeholder(tf.int32, shape=(None, self.max_seq_length))
        self.y_ph = tf.placeholder(tf.int32, shape=(None, ))
        self.seq_length_ph = tf.placeholder(tf.int32, shape=(None, ))
        self.embedding_ph = tf.placeholder(tf.float32, shape=(self.vocab_size, self.word_embedding_size))
        return {'X': self.X_ph, 'y': self.y_ph, 'seq_length': self.seq_length_ph, 'embedding': self.embedding_ph}

    def build_gragh(self):
        word_embeddings = tf.get_variable('word_embeddings_v', shape=(self.vocab_size,
                                    self.word_embedding_size), dtype=tf.float32, trainable=False)
        self.embedding_init = word_embeddings.assign(self.embedding_ph)
        output = tf.nn.embedding_lookup(word_embeddings, self.X_ph)

        with tf.name_scope("Bi-LSTM"):
            for idx in range(self.layer_size):
                with tf.name_scope("Bi-LSTM_%i"%idx):
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_size, state_is_tuple=True, initializer=tf.orthogonal_initializer()),
                        output_keep_prob=self.dropoutKeepProb)

                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_size, state_is_tuple=True, initializer=tf.orthogonal_initializer()),
                        output_keep_prob=self.dropoutKeepProb)

                    outputs_, final_states = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, output,
                            sequence_length=self.seq_length_ph, dtype=tf.float32, scope="bi-lstm_" + str(idx)
                            )

                    if idx < self.layer_size-1:
                        output = tf.concat(outputs_, 2)
                    else:
                        output = tf.concat([final_states[0][1], final_states[1][1]], axis=1)

        # TODO: Attention
        # with tf.name_scope("Attention"):
        #     output = self.__attention(output)
        output = tf.layers.dense(output, 400, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.tanh, name='fc1')
        output = tf.layers.dense(output, 200, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 activation=tf.tanh, name='fc2')
        self.logits = tf.layers.dense(output, 3, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name='final_fc')
        self.y_pred = tf.nn.softmax(self.logits)


    def __attention(self, H):
        pass

if __name__ == '__main__':
    m = Model()