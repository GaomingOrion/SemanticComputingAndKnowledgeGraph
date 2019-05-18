#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
from __future__ import print_function
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation
from neuralnets.BiLSTM import BiLSTM


import _locale
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

modelPath = './models/mydata_0.9568_0.0000_23.h5'
inputPath = 'data/mydata/dev.txt'
outputPath = 'data/mydata/dev_pred.txt'
inputColumns = {1: "tokens"}


# :: Prepare the input ::
sentences = readCoNLL(inputPath, inputColumns)
addCharInformation(sentences)
addCasingInformation(sentences)


# :: Load the model ::
lstmModel = BiLSTM.loadModel(modelPath)


dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


# :: Output to stdout ::
fout = open(outputPath, 'w', encoding='utf-8')
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']

    for tokenIdx in range(len(tokens)):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

        fout.write(str(tokenIdx+1) + '\t' + "%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags))+'\n')
    fout.write('\n')
fout.close()