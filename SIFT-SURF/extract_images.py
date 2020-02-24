import mxnet as mx
import numpy as np
import pickle
import cv2

def extractImagesAndLabels(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f, encoding='bytes')
    images = dict[b'data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict[b'labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray

def extractCategories(path, file):
    f = open(path+file, 'rb')
    dict = pickle.load(f, encoding='bytes')
    return dict[b'label_names']

def saveCifarImage(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1,2,0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path+file+".png", array)

categories = extractCategories("./cifar-10-batches-py/", "batches.meta")

for batch_no in range(5):
    image_array, label_array = extractImagesAndLabels("./cifar-10-batches-py/", "data_batch_"+str(batch_no+1))
    # image_array, label_array = extractImagesAndLabels("./cifar-10-batches-py/", "test_batch")
    # print(image_array.shape)
    # print(label_array.shape)

    
    for var in range(0,10000):
        new_var=(batch_no)*10000+var
        category = label_array[var].asnumpy()
        category = (int)(category[0])
        print(saveCifarImage(image_array[var], "./cifar-10/" + categories[category].decode("utf-8"), "/image"+str(new_var)))

batch_no = 5
# image_array, label_array = extractImagesAndLabels("./cifar-10-batches-py/", "data_batch_"+str(batch_no+1))
image_array, label_array = extractImagesAndLabels("./cifar-10-batches-py/", "test_batch")
# print(image_array.shape)
# print(label_array.shape)


for var in range(0,10000):
    new_var=(batch_no)*10000+var
    category = label_array[var].asnumpy()
    category = (int)(category[0])
    print(saveCifarImage(image_array[var], "./cifar-10/" + categories[category].decode("utf-8"), "/image"+str(new_var)))
