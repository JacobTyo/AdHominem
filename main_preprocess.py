# -*- coding: utf-8 -*-
from helper_functions import Corpus
import os
import pickle
import argparse
"""
    - Dataset can be downloaded here: https://github.com/marjanhs/prnn
    - Pretrained word embeddings (binary file): https://fasttext.cc/docs/en/english-vectors.html
    
"""

parser = argparse.ArgumentParser(description='AdHominem - preprocessing')

parser.add_argument('-dataset', default='amazon', type=str)  # character embedding dimension
parser.add_argument('-train_path', default='/home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/train.csv', type=str)
parser.add_argument('-test_path', default='/home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/test.csv', type=str)
parser.add_argument('-embeddings', default='fasttext', type=str)
parser.add_argument('-save_path', default=os.path.join('data', 'amazon.pkl'), type=str)
# '/home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/train_test_adhominem_glove.pkl'
parser.add_argument('-distributed', action='store_true')
args = parser.parse_args()
hyper_parameters = vars(args)

# for gutenburg corpus
corpus = Corpus(dataset=args.dataset, train_path=args.train_path,
                test_path=args.test_path,
                embeddings=args.embeddings)
if args.distributed:
    corpus.extract_docs()
else:
    corpus.extract_docs_single_thread()
corpus.remove_rare_tok_chr()
corpus.make_wrd_chr_vocabularies()

with open(args.save_path, 'wb') as f:
    pickle.dump((corpus.docs_L_tr, corpus.docs_R_tr, corpus.labels_tr,
                 corpus.docs_L_te, corpus.docs_R_te, corpus.labels_te,
                 corpus.V_w, corpus.E_w, corpus.V_c), f)
