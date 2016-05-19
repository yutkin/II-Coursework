from __future__ import print_function
import numpy as np
import scipy as sp
import os
import cPickle

class CIFAR10(object):
        def __init__(self, path):
        self.path = path
        f_meta = open(os.path.join(self.path, 'batches.meta'), 'r')
        self.meta = cPickle.load(f_meta)
        f_meta.close()
    def get_train_data(self):
        data, labels, fnames = [], [], []
        for i in xrange(1,6):
            f_batch = open(os.path.join(self.path, 'data_batch_%d' % i), 'r')
            batch = cPickle.load(f_batch)
            f_batch.close()
            for j in range(len(batch['filenames'])):
                data.append(np.array(batch['data'][j],
                                     dtype=float).reshape((3, 32, 32)))
                labels.append(batch['labels'][j])
                fnames.append(batch['filenames'][j])
        return np.array(data), np.array(labels), np.array(fnames)
    
    def get_test_data(self):
        data, labels, fnames = [], [], []
        f_batch = open(os.path.join(self.path, 'test_batch'), 'r')
        batch = cPickle.load(f_batch)
        f_batch.close()
        for i in range(len(batch['filenames'])):
            data.append(np.array(batch['data'][i],
                                 dtype=float).reshape((3, 32, 32)))
            labels.append(batch['labels'][i])
            fnames.append(batch['filenames'][i])
        return np.array(data), np.array(labels), np.array(fnames)
    
    def get_whitened_data(self, alpha, kind='train'):
        if kind == 'train':
            data, labels, fnames = self.get_train_data()
        elif kind == 'test':
            data, labels, fnames = self.get_test_data()
        else:
            raise RuntimeError('"%s" is not acceptible. '
                               'Either "test" or "train"' % kind)
        # ZCA
        N = data.shape[0]
        X = data.reshape((N,-1)) / 255
        X -= np.mean(X, axis=0)
        sigma = np.dot(X.T, X) / N
        U, S, V = np.linalg.svd(sigma)
        sqS = np.sqrt(S + alpha)
        Uzca = np.dot(U / sqS[np.newaxis, :], U.T)
        Z = np.dot(X, Uzca.T)
        zca_data = []
        for i in xrange(N):
            absmax = np.max(np.abs(Z[i, :]))
            tmp = Z[i, :].reshape((3, 32, 32)) / absmax*127 + 128
            zca_data.append(np.array(tmp.transpose(1,2,0), dtype=np.uint8))
        return np.array(zca_data), labels, fnames
