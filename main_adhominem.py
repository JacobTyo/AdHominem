# -*- coding: utf-8 -*-
from adhominem import AdHominem
from adhominem_noCNN import AdHominem_NoCNN
from adhominem_noFastText import AdHominem_NoFastText
from adhominem_flat import AdHominem_flat
from adhominem_noCNN_flat import AdHominem_NoCNN_flat
import pickle
import os
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim

import mlflow


def main(hyper_parameters):

    os.environ["CUDA_VISIBLE_DEVICES"] = hyper_parameters['device']

    # create folder for results
    # dir_results = os.path.join('results')
    # if not os.path.exists(dir_results):
    #     os.makedirs(dir_results)

    # load docs, vocabularies and initialized word embeddings
    with open(hyper_parameters['vocab_wordemb_file'], 'rb') as f:
        docs_L_tr, docs_R_tr, labels_tr, \
        docs_L_te, docs_R_te, labels_te, \
        V_w, E_w, V_c = pickle.load(f)

    # add vocabularies to dictionary
    hyper_parameters['V_w'] = V_w
    hyper_parameters['V_c'] = V_c
    hyper_parameters['N_tr'] = len(labels_tr)
    hyper_parameters['N_dev'] = len(labels_te)

    if not hyper_parameters['no_mlflow']:
        for k in ['V_c', 'V_w']:
            mlflow.log_param(k, len(hyper_parameters[k]))
        for k in ['N_tr', 'N_dev']:
            mlflow.log_param(k, hyper_parameters[k])

    # load neural network model
    if hyper_parameters['no_cnn'] and hyper_parameters['flatten']:
        adhominem = AdHominem_NoCNN_flat(hyper_parameters=hyper_parameters,
                                         E_w_init=E_w
                                         )
    elif hyper_parameters['no_cnn']:
        adhominem = AdHominem_NoCNN(hyper_parameters=hyper_parameters,
                                    E_w_init=E_w
                                    )
    elif hyper_parameters['no_fasttext']:
        adhominem = AdHominem_NoFastText(hyper_parameters=hyper_parameters,
                                         E_w_init=E_w
                                         )
    elif hyper_parameters['flatten']:
        adhominem = AdHominem_flat(hyper_parameters=hyper_parameters,
                                   E_w_init=E_w
                                   )
    else:
        adhominem = AdHominem(hyper_parameters=hyper_parameters,
                              E_w_init=E_w
                              )
    # start training
    train_set = (docs_L_tr, docs_R_tr, labels_tr)
    test_set = (docs_L_te, docs_R_te, labels_te)


    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    adhominem.train_model(train_set, test_set)

    # close session
    adhominem.sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AdHominem - Siamese Network for Authorship Verification')
    parser.add_argument('-D_c', default=10, type=int)  # character embedding dimension
    parser.add_argument('-D_r', default=30, type=int)  # character representation dimension
    parser.add_argument('-w', default=4, type=int)  # length of 1D-CNN sliding window
    parser.add_argument('-D_w', default=300, type=int)  # dimension of word embeddings
    parser.add_argument('-D_s', default=50, type=int)  # dimension sentence embeddings
    parser.add_argument('-D_d', default=50, type=int)  # dimension of document embedding
    parser.add_argument('-D_mlp', default=60, type=int)  # final output dimension
    parser.add_argument('-T_c', default=15, type=int)  # maximum number of characters per words
    parser.add_argument('-T_w', default=20, type=int)  # maximum number of words per sentence
    parser.add_argument('-T_s', default=50, type=int)  # maximum number of sentences per document
    parser.add_argument('-t_s', default=0.95, type=float)  # boundary for similar pairs (close to one)
    parser.add_argument('-t_d', default=0.05, type=float)  # boundary for dissimilar pairs (close to zero)
    parser.add_argument('-epochs', default=100, type=int)  # total number of epochs
    parser.add_argument('-train_word_embeddings', default=False, type=bool)  # fine-tune pre-trained word embeddings
    parser.add_argument('-batch_size_tr', default=32, type=int)  # batch size for training
    parser.add_argument('-batch_size_te', default=128, type=int)  # batch size for evaluation
    parser.add_argument('-initial_learning_rate', default=0.002, type=float)  # initial learning rate
    parser.add_argument('-keep_prob_cnn', default=0.9, type=float)  # dropout for 1D-CNN layer
    parser.add_argument('-keep_prob_lstm', default=0.9, type=float)  # variational dropout for BiLSTM layer
    parser.add_argument('-keep_prob_att', default=0.9, type=float)  # dropout for attention layer
    parser.add_argument('-keep_prob_metric', default=0.9, type=float)  # dropout for metric learning layer
    parser.add_argument('-results_file', default='results.txt', type=str)
    parser.add_argument('-loss', default='modified_contrastive', type=str)
    parser.add_argument('-no_cnn', action='store_true')
    parser.add_argument('-no_fasttext', action='store_true')
    parser.add_argument('-vocab_wordemb_file', default='data/amazon.pkl', type=str)
    # /home/jtyo/Repos/AuthorshipAttribution/data/_gutenburg/train_test_adhominem.pkl
    parser.add_argument('-device', default='0', type=str)
    parser.add_argument('-flatten', action='store_true')
    parser.add_argument('-no_mlflow', action='store_true')
    parser.add_argument('-experiment_name', default='default', type=str)
    parser.add_argument('-run_name', default='default', type=str)
    args = parser.parse_args()


    if args.no_mlflow:
        main(vars(args))

    else:
        mlflow.set_tracking_uri('http://gs18196.sp.cs.cmu.edu:6460')
        mlflow.set_experiment(args.experiment_name)

        with mlflow.start_run(run_name=args.run_name):

            # log all of the args to mlflow
            try:
                for key, val in vars(args).items():
                    mlflow.log_param(key, val)
            except Exception as e:
                for key, val in args.items():
                    mlflow.log_param(key, val)

            main(vars(args))
