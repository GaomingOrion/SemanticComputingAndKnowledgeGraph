import tensorflow as tf
import numpy as np
from sklearn.metrics import recall_score, f1_score
from model import Model
from Dataset import Dataset, get_embeddings

class TwitterSem:
    def __init__(self, prev_model=None):
        self.epochs = 200
        self.prev_model = prev_model
        self.train_data = Dataset('train', 64)
        self.dev_data = Dataset('dev', 100)
        self.test_data = Dataset('test', 100)
        self.model = Model()

    def train(self):
        total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.model.y_ph, logits=self.model.logits))
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        # optimizer = tf.train.AdamOptimizer(0.0001)
        train_op = optimizer.minimize(total_loss)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            if self.prev_model:
                saver.restore(sess, self.prev_model)
            else:
                sess.run(tf.global_variables_initializer())
                emb_matrix, _ = get_embeddings()
                sess.run(self.model.embedding_init, feed_dict={self.model.placeholders['embedding']: emb_matrix})
            for epoch in range(1, self.epochs+1):
                for X, y, seq_length in self.test_data.one_epoch_generator():
                    feed_dict = {
                        self.model.placeholders['X']: X,
                        self.model.placeholders['y']: y,
                        self.model.placeholders['seq_length']: seq_length
                    }
                    sess.run(train_op, feed_dict=feed_dict)
                print('--------开始评测--------')
                print('loss: ', sess.run(total_loss, feed_dict=feed_dict))
                print('epoch: %i'%epoch)
                print('(Average_recall, F1_PN, Accuarcy)')
                print('dev_set result: ', self.evaluate(sess, mod='dev'))
                print('test_set result: ', self.evaluate(sess, mod='test'))
                print('--------评测结束--------\n')

    def evaluate(self, sess, mod):
        assert mod in ['dev', 'test']
        dataset = self.dev_data if mod == 'dev' else self.test_data
        y_preds = []
        for X, y, seq_length in dataset.one_epoch_generator():
            feed_dict = {
                self.model.placeholders['X']: X,
                self.model.placeholders['y']: y,
                self.model.placeholders['seq_length']: seq_length
            }
            y_preds.append(sess.run(self.model.y_pred, feed_dict=feed_dict))
        y_preds = np.concatenate(y_preds)
        if mod == 'test':
            with open('./data/y_test_pred.txt', 'w') as f:
                for x in np.argmax(y_preds, 1):
                    f.write(str(x) + '\n')
        return self.metric(dataset.y, y_preds)

    def metric(self, y_true, y_preds):
        y_preds_label = np.argmax(y_preds, axis=1)
        acc = np.sum(y_true ==y_preds_label)/y_true.shape[0]
        avg_rec = recall_score(y_true, y_preds_label, average='macro')
        f1_pn = f1_score(y_true, y_preds_label, average='macro', labels=[0, 2])
        return avg_rec, f1_pn, acc


if __name__ == '__main__':
    a = TwitterSem()
    a.train()