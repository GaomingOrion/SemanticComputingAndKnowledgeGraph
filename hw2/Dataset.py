import numpy as np
import pickle

class Dataset:
    def __init__(self, mod, batch_size):
        self.dataset_name = mod
        self.max_seq_length = 30
        self.batch_size = batch_size
        _, self.word_dict = get_embeddings()
        assert mod in ['train', 'dev', 'test']
        self.x_path = './data/preprocessed/x_' + mod + '.txt'
        self.y_path = './data/preprocessed/y_' + mod + '.txt'
        self.X, self.y, self.seq_length = self.__load_data()
        assert len(self.X) == len(self.y)
        self.size = len(self.X)

    def __load_data(self):
        X, seq_length = [], []
        with open(self.x_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                res, length = self.__embedding_one_seq(line.rstrip().split())
                X.append(res)
                seq_length.append(length)
        with open(self.y_path, 'r') as f:
            y = list(map(lambda s: int(s.rstrip()), f.readlines()))
        return np.int32(X), np.int32(y), np.int32(seq_length)

    def __embedding_one_seq(self, seq):
        res = [0]*self.max_seq_length
        seq_length = min(self.max_seq_length, len(seq))
        for i in range(min(self.max_seq_length, len(seq))):
            res[i] = self.word_dict.get(seq[i], self.word_dict['<unknown>'])
        return res, seq_length

    def one_epoch_generator(self):
        idx = list(range(self.size))
        if self.dataset_name == 'train':
            np.random.shuffle(idx)
        start = 0
        while start < self.size:
            end = start + self.batch_size
            yield self.X[start:end], self.y[start:end], self.seq_length[start:end]
            start = end


def get_embeddings(init=False, dim=200):
    if init:
        word_dict = dict()
        emb_matrix = []
        with open('./data/worddata/glove.twitter.27B.200d.txt', encoding='utf8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.split(' ')
                word_dict[line[0]] = i
                emb_matrix.append(list(map(float, line[1:])))
        vectors = np.float32(emb_matrix)
        with open('./data/worddata/word_dict.pkl', "wb") as f:
            pickle.dump(word_dict, f)
        with open('./data/worddata/embedding_matrix.pkl', "wb") as f:
            pickle.dump(vectors, f)
    else:
        with open('./data/worddata/word_dict.pkl', "rb") as f:
            word_dict = pickle.load(f)
        with open('./data/worddata/embedding_matrix.pkl', "rb") as f:
            emb_matrix = pickle.load(f)
    return emb_matrix, word_dict

if __name__ == '__main__':
    #
    #get_embeddings(init=True)
    d = Dataset('train', 2)