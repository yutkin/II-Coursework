from __future__ import print_function, division
import caffe
import pandas as pd
import numpy as np
import lmdb

MODEL_PROTO = '/data/deploy.prototxt'
MODEL_WEIGHTS = '/data/_iter_70000.caffemodel'
MEAN = '/data/cifar10zca_mean.binaryproto'

TRAIN_LMDB = '/data/cifar10zca_train_lmdb'
TEST_LMDB = '/data/cifar10zca_test_lmdb'

caffe.set_device(0)
caffe.set_mode_gpu()

# Load CNN
net = caffe.Net(MODEL_PROTO, MODEL_WEIGHTS, caffe.TEST)

# Getting mean
mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
f = open(MEAN, 'rb')
mean_blobproto_new.ParseFromString(f.read())
mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
f.close()

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship','truck']

# Preparing database
lmdb_cursor = lmdb.open(TRAIN_LMDB).begin().cursor()

answers = []
guesed = 0
for i, key, value in enumerate(lmdb_cursor):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(value)
    label = int(datum.label)
    image = caffe.io.datum_to_array(datum).astype(np.uint8)
    image = np.asarray([image]) - mean_image
    image = image[:, :, 4:36, 4:36]

    forward_pass = net.forward_all(data=image)
    probabilities = forward_pass['prob'][0]
    answers.append(np.append(probabilities, label))

    predicted = probabilities.argmax()
    guesed += int(label == predicted)

print('Model accuracy: ', guesed / i)

# Save model answers to csv
model = pd.DataFrame(answers, columns=labels + ['ground truth'])
model.to_csv('model_train_answers.csv', index=False)


