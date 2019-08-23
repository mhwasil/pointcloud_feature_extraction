#!/usr/bin/env python

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath('__file__'))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'fisher_vector'))
import utils

import sys
import os.path
import argparse
from svm_trainer import SVMTrainer
import warnings

#  ./train_classifier.py --dataset objects_daylight --objects F20_20_G S40_40_G --output <name of classifier>
#  ./train_classifier.py --dataset objects_daylight --output <name of classifier>

if __name__ == '__main__':
    
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    parser = argparse.ArgumentParser(description='''
    Train and save SVM classifier for object recognition.
    The classifier will be saved to the "common/config/" folder.
    ''')
    parser.add_argument('--dataset', help='dataset to use',
                        default='/media/emha/HDD4/pointcloud_dataset/pcd/asus/nagoya_montreal')
                        # /home/emha/pmd_multiview_dataset/
                        # /media/emha/HDD1/Dataset/rgbd-dataset-51/
    parser.add_argument('--objects', help='list of objects to use',
                        nargs='*', default='all')

    # mc, msc_fv_2,msc_fv_3,msc_fv_5, fv_2, fv_3, fv_5
    parser.add_argument('--output', help='output filename (without folder and '
                        'extension, default: "classifier")', default='fv_3')
    args = parser.parse_args()
    
    data_folder =  args.dataset
    cfg_folder = './config/maxrdd_fv_asus'

    # cls_names = ['mc', 'fv_2','fv_3','fv_4','fv_5','msc_fv_2','msc_fv_3', 'msc_fv_4', 'msc_fv_5']
    # n_gauss = [1,2,3,4,5,2,3,4,5]
    #cls_names = ['mc']
    cls_names = ['msc_fv_3']
    #cls_names = ['mc', 'fv_2','fv_3','msc_fv_2','msc_fv_3', 'msc_fv_4']

    for i,cls_name in enumerate(cls_names):
        print ("Classififer: ", cls_name)
        if cls_name == "mc":
            n_gaussian = 1
        else:
            n_gaussian = int(''.join(filter(str.isdigit, cls_name)))

        gmm = utils.get_3d_grid_gmm(subdivisions=[n_gaussian, n_gaussian, n_gaussian], variance=0.05)

        trainer = SVMTrainer(data_folder)
        #classifier = trainer.train_using_pgz(data_folder, gmm, feature_extraction=cls_name,datasize=10)
        classifier = trainer.train(data_folder, 'all', gmm, feature_extraction=cls_name, mc_feature=True)

        directory = os.path.join(cfg_folder, cls_name)
        print ("Saving classifer to ", directory)
        if not os.path.exists(directory):
            print ("Creating directory ", directory)
            os.makedirs(directory)
        classifier.save(os.path.join(cfg_folder, cls_name, 'classifier.pkl'),
                        os.path.join(cfg_folder, cls_name, 'label_encoder.pkl'))
