#!/usr/bin/env python

import sys
from os.path import join
import numpy as np
import argparse
import glob
import os
import re

import utils

# Import helper class for loading trained network
from svm_classifier import SVMObjectClassifier
from features import calculate_feature_vector, calculate_maxrdd_fv_features
from svm_trainer import SVMTrainer
#import open3d
import pcl
import gzip
import pickle

import struct
import colorsys
import warnings

import matplotlib.pyplot as plt

def parse_pcd(input_file, enable_color=False):
    file_data = pcl.load(input_file).to_array()
    n_points, n_dims = file_data.shape
    if enable_color is True:
        color_table = np.float64(file_data[:, 3:])
        color_table = np.array([list(colorsys.rgb_to_hsv(i[0]/255.0, i[1]/255.0, i[2]/255.0)) for i in color_table])
        point_cloud = np.hstack([file_data[:, 0:3], color_table])
    else:
        point_cloud = file_data[:, 0:3]

    return point_cloud

# copied from python-pcl
def float_to_rgb(p_rgb):
    # rgb = *reinterpret_cast<int*>(&p.rgb)
    rgb_bytes = struct.pack('f', p_rgb)
    rgb = struct.unpack('I', rgb_bytes)[0]
    r = (rgb >> 16) & 0x0000ff
    g = (rgb >> 8)  & 0x0000ff
    b = (rgb)       & 0x0000ff
    return (r/255.0),(g/255.0),(b/255.0)

def load_classifier(classifier_name):
    cfg_folder = 'config'
    return SVMObjectClassifier.load(join(cfg_folder, classifier_name, 'classifier.pkl'),
                                    join(cfg_folder, classifier_name, 'label_encoder.pkl'))

# def convert_to_xyzhsv(pc):
#     # Generator for x,y,z,rgb fields from pointcloud
#     #cloud = open3d.read_point_cloud(pc)
#     xyzrgb_gen = parse_pcd(input_file=pc, enable_color=True)

#     #xyzrgb_gen = sensor_msgs.point_cloud2.read_points(pc, skip_nans=False, field_names=("x", "y", "z", "rgb"))

#     # convert generator to list of lists then numpy array
#     xyzrgb = [list(elem) for elem in list(xyzrgb_gen)]
#     xyzrgb = np.array(xyzrgb)
#     rgb = xyzrgb[:,3][np.newaxis].T
    
#     xyz = np.asarray(cloud.points)
#     rgb = np.asarray(cloud.colors)
#     rgb = rgb[np.newaxis].T
#     print (rgb.shape)
#     hsv = np.array([list(colorsys.rgb_to_hsv(*float_to_rgb(i))) for i in rgb])
#     xyzhsv = np.hstack([xyzrgb[:,0:3], hsv])

#     return xyzhsv

def evaluate_cls(data_folder, classifier, objects='all', color=False):
    objects_to_test = os.listdir(os.path.abspath(data_folder))
    print ("Training classifer for objects: ", objects_to_test)

    seen = np.zeros((len(objects_to_test)))
    correct = np.zeros((len(objects_to_test)))

    for i,obj in enumerate(objects_to_test):
        class_folder = os.path.join(data_folder, obj, "")
        files =  glob.glob(class_folder + '**/*.pcd', recursive=True)
        for f in files:
            #xyzhsv = convert_to_xyzhsv(f)
            xyzhsv = parse_pcd(f)
            features = calculate_feature_vector(xyzhsv, False)
            features = np.reshape(features, (1,-1))
            label, probability = classifier.classify(features)
            seen[i] += 1
            if label == str(obj):
                correct[i] += 1

    print ("Seen: ",seen)
    print ("Correct: ", correct)
    print ("Avg class acc: ", correct/seen)
    print ("Accuracy: ", np.mean(correct, axis=0))

def evaluate_maxrdd_fv_cls(data_folder, classifier, objects='all', color=False, 
                            gmm=None, feature_extraction='msc_fv_3', mc_feature=True):

    objects_to_eval = []
    object_directories = np.array(glob.glob(data_folder + '/*'))
    for obj_dir in object_directories:
        print (obj_dir)
        object_name = obj_dir.split('/')[-1]
        objects_to_eval.append(str(object_name))
    seen = np.zeros((len(objects_to_eval)))
    correct = np.zeros((len(objects_to_eval)))
    print (objects_to_eval)
    for i,obj in enumerate(objects_to_eval):
        files = np.array(glob.glob(data_folder + '/' + obj + '/*'))
        for f in files:
            points = parse_pcd(f, enable_color=True)
            if feature_extraction == 'mc':
                features = calculate_feature_vector(points, enable_color=True)
            else:
                features = calculate_maxrdd_fv_features(points, gmm, feature_extraction, mean_circle=mc_feature)
            #xyzhsv = parse_pcd(f)
            #features = calculate_feature_vector(xyzhsv, False)
            features = np.reshape(features, (1,-1))
            print ("Shape: ",features.shape)
            label, probability = classifier.classify(features)
            seen[i] += 1
            if label == str(obj):
                correct[i] += 1
            print ("True: {}, Pred: {}".format(str(obj), label))

    print ("Seen: ",seen)
    print ("Correct: ", correct)
    print ("Avg class acc: ", correct/seen)
    print ("Accuracy: ", np.sum(correct/seen)/len(objects_to_eval))

    return  np.mean(np.sum(correct/seen)/len(objects_to_eval))

def load_compressed_pickle_file(pickle_file_name):
    with gzip.open(pickle_file_name, 'rb') as f:
        return pickle.load(f)

def evaluate_cls_using_pgz(data_folder, gmm, classifier, feature_extraction='mc', idx=[5,6],objects='all', color=False):
    data = None
    labels = None
    count = 0
    for i in range(idx[0],idx[1]):
        print ("Loading:...",'{}data{}.pgz'.format(data_folder,i))
        dataset = load_compressed_pickle_file('{}data{}.pgz'.format(data_folder,i))
        print (dataset['data'].shape)
        if count == 0:
            data = dataset['data']
            labels = dataset['labels']
        else:
            data = np.concatenate((data, dataset['data']))
            labels = np.concatenate((labels, dataset['labels']))
        count += 1

    print (labels.shape)

    seen = np.zeros((len(labels)))
    correct = np.zeros((len(labels)))

    for i,points in enumerate(data):
        #print ("ITERA",i)
            #files = np.array(glob.glob(self.data_folder + '/' + obj + '/*'))
        if feature_extraction == 'mc':
            features = calculate_feature_vector(points, False)
        else:
            features = calculate_maxrdd_fv_features(points, gmm, feature_extraction)
        #features = np.nan_to_num(features)
        features = np.reshape(features, (1,-1))
        pred, probability = classifier.classify(features)
        seen[i] += 1
        if pred == labels[i]:
            correct[i] += 1
        print ("True: {}, Pred: {}".format(pred, labels[i]))

    print ("Seen: ",seen)
    print ("Correct: ", correct)
    print ("Avg class acc: ", correct/seen)
    print ("Accuracy: ", np.mean(correct, axis=0))

    return  np.mean(correct, axis=0)


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    f = ['msc_fv_3']#,'fv_3','fv_4','fv_5','msc_fv_2','msc_fv_3', 'msc_fv_4', 'msc_fv_5']
    accuracies = []
    indices = []
    #f = ['mc', 'fv_2','fv_3','msc_fv_2','msc_fv_3', 'msc_fv_4']
    for idx,fe in enumerate(f):
        print ("==========================")
        print ("Feature extraction: ",fe)
        #classifier = load_classifier('mc_fv_rgbd_washington/{}'.format(fe))
        classifier = load_classifier('maxrdd_fv_asus/{}'.format(fe))
        print (classifier.label_encoder.classes_)

        parser = argparse.ArgumentParser(description='''
        Train and save SVM classifier for object recognition.
        The classifier will be saved to the "common/config/" folder.
        ''')
        parser.add_argument('--pcd', help='cloud to test',
                            default='data/test/PMD_M20.pcd')
        parser.add_argument('--dataset', help='cloud to test',
                            default='/media/emha/HDD4/pointcloud_dataset/pcd/asus/test')
                            # /home/emha/pmd_multiview_dataset/
                            # /media/emha/HDD1/Dataset/rgbd-dataset-51/
        args = parser.parse_args()
        cloud =  args.pcd
        data_folder = args.dataset
        gmm = None
        if fe is not "mc":
            n_gaussian = int(''.join(filter(str.isdigit, fe)))
            print (n_gaussian)
            gmm = utils.get_3d_grid_gmm(subdivisions=[n_gaussian, n_gaussian, n_gaussian], variance=0.05)

        acc = evaluate_maxrdd_fv_cls(data_folder, classifier, objects='all', color=True, 
                            gmm=gmm, feature_extraction=fe, mc_feature=True)
        
        #acc = evaluate_cls_using_pgz(data_folder, gmm, classifier, fe, [10,13])
        accuracies.append(acc)
        indices.append(int(idx))

    plt.scatter(indices,accuracies)
    plt.xticks(indices, f)
    plt.grid()
    plt.show()