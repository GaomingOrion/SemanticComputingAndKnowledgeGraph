import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
import gensim

class WordSimilarity:
    def __init__(self):
        self.MTURK_path = './data/MTURK-771.csv'
        self.scores = pd.read_csv(self.MTURK_path, header=None)
        self.scores.columns = ['word1', 'word2', 'human_score']
        self.word2vec_model_path = 'F:\\DATA\\word_embedding\\GoogleNews-vectors-negative300.bin'
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.word2vec_model_path, binary=True)

    def _wordnet_score(self, word1, word2):
        res = []
        for w1 in wn.synsets(word1):
            for w2 in wn.synsets(word2):
                try:
                    s = w1.lch_similarity(w2)
                    if s:
                        res.append(s)
                except:
                    pass
        return np.max(res)

    def cal_wordnet_score(self):
        self.scores['wordnet_score'] = self.scores.apply(lambda x: self._wordnet_score(x['word1'], x['word2']), axis=1)

    def _word2vec_score(self, word1, word2):
        return self.word2vec_model.similarity(word1, word2)

    def cal_word2vec_score(self):
        self.scores['word2vec_score'] = self.scores.apply(lambda x: self._word2vec_score(x['word1'], x['word2']), axis=1)

    def main(self):
        self.cal_wordnet_score()
        self.cal_word2vec_score()
        print(self.scores.corr(method='pearson'))


if __name__ == '__main__':
    a = WordSimilarity()
    a.main()
    # a.scores.loc[:, ['human_score', 'wordnet_score', 'word2vec_score']] = a.scores.loc[:, ['human_score', 'wordnet_score', 'word2vec_score']].apply(
    #     lambda x: (x-np.mean(x))/np.sqrt(np.var(x))
    # )
    a.scores.to_csv('./data/result.csv')