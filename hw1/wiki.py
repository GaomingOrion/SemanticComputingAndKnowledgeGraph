from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus, MmCorpus

wiki_bz_path = 'F:\\DATA\\wiki\\enwiki-20190301-pages-articles.xml.bz2'
path_to_wiki_dump = datapath(wiki_bz_path)
corpus_path = get_tmpfile("F:\\DATA\\wiki\\wiki-corpus.mm")
wiki = WikiCorpus(path_to_wiki_dump)  # create word->word_id mapping, ~8h on full wiki
MmCorpus.serialize(corpus_path, wiki)