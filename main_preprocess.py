# -*- coding: utf-8 -*-
from helper_functions import Corpus
import os
import pickle
"""
    - Dataset can be downloaded here: https://github.com/marjanhs/prnn
    - Pretrained word embeddings (binary file): https://fasttext.cc/docs/en/english-vectors.html
    
"""

# for gutenburg corpus
corpus = Corpus(dataset='gutenburg', train_path='/home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/train.csv',
                test_path='/home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/test.csv',
                embeddings='glove')
corpus.extract_docs()
corpus.remove_rare_tok_chr()
corpus.make_wrd_chr_vocabularies()

with open('/home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/train_test_adhominem_glove.pkl', 'wb') as f:
    pickle.dump((corpus.docs_L_tr, corpus.docs_R_tr, corpus.labels_tr,
                 corpus.docs_L_te, corpus.docs_R_te, corpus.labels_te,
                 corpus.V_w, corpus.E_w, corpus.V_c), f)
