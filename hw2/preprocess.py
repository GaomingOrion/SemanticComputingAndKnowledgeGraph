import preprocessor as p
#p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.SMILEY, p.OPT.NUMBER)
trans_dict = {'$HASHTAG$':'<hashtag>', '$URL$':'<url>', '$MENTION$':'<user>', '$SMILEY$':'<smile>', '$NUMBER$':'<number>'}
import html
import random
import os

class Preprocess:
    def __init__(self):
        self.data_dir = './data/'
        self.out_dir = './data/preprocessed/'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        self.sem_dict = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __clean_text(self, raw_text):
        text = raw_text.rstrip()
        if '""' in text:
            if text[0] == text[-1] == '"':
                text = text[1:-1]
            text = text.replace('\\""', '"')
            text = text.replace('""', '"')
        text = text.replace('\\""', '"')
        text = html.unescape(text)
        text = ' '.join(text.split())
        return text

    def __transform(self, raw_text):

        return p.clean(raw_text.lower())

    def create_test_data(self):
        data_path = self.data_dir+'test/'
        fx = open(self.out_dir+'x_test.txt', 'w', encoding='utf-8')
        fy = open(self.out_dir+'y_test.txt', 'w', encoding='utf-8')
        for path in os.listdir(data_path):
            print(path)
            fin = open(data_path+path, encoding='utf-8')
            for i, line in enumerate(fin.readlines()):
                if (i+1)%1000 == 0:
                    print(i+1)
                _, sem, raw_text = line.split('\t')
                sem_idx = self.sem_dict[sem]
                text = self.__transform(raw_text)
                fx.write(text + '\n')
                fy.write(str(sem_idx) + '\n')
            fin.close()
        fx.close()
        fy.close()

    def create_train_dev_data(self):
        data_path = self.data_dir+'train/'
        train_cnt, dev_cnt = 0, 0
        fxtrain = open(self.out_dir+'x_train.txt', 'w', encoding='utf-8')
        fytrain = open(self.out_dir+'y_train.txt', 'w', encoding='utf-8')
        fxdev = open(self.out_dir+'x_dev.txt', 'w', encoding='utf-8')
        fydev = open(self.out_dir+'y_dev.txt', 'w', encoding='utf-8')
        for path in os.listdir(data_path):
            print(path)
            fin = open(data_path+path, encoding='utf-8')
            for i, line in enumerate(fin.readlines()):
                if (i+1)%1000 == 0:
                    print(i+1)
                try:
                    split_res = line.strip().split('\t')
                    sem = split_res[1]
                    raw_text = ' '.join(split_res[2:])
                except:
                    pass
                sem_idx = self.sem_dict[sem]
                text = self.__transform(raw_text)
                if random.random() < 0.9:
                    train_cnt += 1
                    fxtrain.write(text + '\n')
                    fytrain.write(str(sem_idx) + '\n')
                else:
                    dev_cnt += 1
                    fxdev.write(text + '\n')
                    fydev.write(str(sem_idx) + '\n')
            fin.close()
        fxtrain.close()
        fytrain.close()
        fxdev.close()
        fydev.close()

if __name__ == '__main__':
    pre = Preprocess()
    print('start preprocessing training data')
    pre.create_train_dev_data()
    print('start preprocessing testing data')
    pre.create_test_data()
