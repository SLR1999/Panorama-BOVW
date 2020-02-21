import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.model_selection import StratifiedKFold   

def features(image, extractor):
    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors

def read_images(folders):
    data = {}
    for folder in folders:
        images = []
        for filename in os.listdir(folder):
            image = cv2.imread(os.path.join(folder, filename))
            if image is not None:
                images.append(image)
        images = np.array(images)
        data[folder] = images
    return data

def create_descriptors(data):
    descriptor_list = []
    image_descriptor = {}
    for class_label in data:
        class_descriptor_list = []
        for image in data[class_label]:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            keypoint, descriptor = features(gray_image, extractor)
            descriptor_list.extend(descriptor)
            class_descriptor_list.append(descriptor)
        class_descriptor_list = np.array(class_descriptor_list)
        image_descriptor[class_label] = class_descriptor_list
    descriptor_list = np.array(descriptor_list)
    return descriptor_list, 42

def build_bag_of_visual_words(model, image_descriptor):
    BoVW = {}
    hist_length = model.get_params()['n_clusters']

    for class_label in image_descriptor:
        
        list_of_histograms = []
        for descriptor in image_descriptor[class_label]:
            hist = [0] * hist_length
            v = model.predict(descriptor.reshape(1, -1))[0]
            hist[v] = hist[v]+1
            list_of_histograms.append(hist)
        BoVW[class_label] = list_of_histograms

    return BoVW
    
extractor = cv2.xfeatures2d.SIFT_create()

data = read_images(["Bikes", "Horses"])
descriptor_list, image_descriptor = create_descriptors(data)

n_clusters = 100

kmeans = KMeans(n_clusters = n_clusters)
kmeans.fit(descriptor_list) 

BoVW = build_bag_of_visual_words(kmeans, image_descriptor)

X = []
y = []

for class_label in BoVW:
    for histogram in BoVW[class_label]:
        X.append(histogram)
        y.append(class_label)

X = np.asarray(X)
y = np.asarray(y)

md = svm.SVC(kernel='linear')
cv = StratifiedKFold(n_splits=5, random_state=42)

scores = []
for train_ind, validate_ind in cv.split(X, y):
    train_X, train_y = X[train_ind], y[train_ind]
    validate_X, validate_y = X[validate_ind], y[validate_ind]
    md.fit(train_X, train_y)
    score = md.score(validate_X, validate_y)
    scores.append(score)

sc = np.array(scores)
print(sc)
print("Score: " + str(np.mean(sc)))
