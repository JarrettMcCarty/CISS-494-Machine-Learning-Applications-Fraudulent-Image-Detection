import os
import shutil
import numpy as numpy
from tqdm import tqdm_notebook, tqdm
import pandas

class PathNotFoundException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None
    
    def __str__(self):
        if self.message:
            return 'PathNotFoundException: {0}'.format(self.message)
        else:
            return "PathNotFoundException raised\n"

class FraudulentImageCleaner(object):

    def __init__(self, controller = "dataset-dist/phase-01/training/"):
        ''' Set up directory variables for cleaning '''
        self.controller = controller
        self.fraud_path = self.controller + "fake/"
        self.clean_path = self.controller + "pristine/"
        self.mask_path = self.fraud_path + "masks/"
        self.frauds = []
        self.cleans = []
        self.masks = []

    ''' Establish how many images reside in each folder '''
    def get_num_frauds(self):
        #print('Number of fake images = {}'.format((len(os.listdir('dataset-dist/phase-01/training/fake'))-1)/2))
        return (len(os.listdir('dataset-dist/phase-01/training/fake'))-2)

    def get_num_clean(self):
        #print('Number of pristine images = {}'.format((len(os.listdir('dataset-dist/phase-01/training/pristine/')))))
        return (len(os.listdir('dataset-dist/phase-01/training/pristine/')))

    def set_frauds(self):
        self.frauds = os.listdir(self.fraud_path)[1:-1] # contains both masked images as well as altered images one after the other

    def set_cleans(self):
        self.cleans = os.listdir(self.clean_path) # all prestine images

    def set_masks(self):
        self.masks = os.listdir(self.mask_path) # all masks

    def get_frauds(self):
        return self.frauds

    def get_cleans(self):
        return self.cleans

    def get_masks(self):
        return self.masks

    def get_fraud_path(self):
        return self.fraud_path

    def get_clean_path(self):
        return self.clean_path

    def get_mask_path(self):
        return self.mask_path

    def seperate_masks(self):
        if not os.path.isdir(self.mask_path):
            os.mkdir(self.mask_path)
            for f in self.frauds:
                if len(f.split(".")) == 3:
                    shutil.move(self.fraud_path + f, self.mask_path)
        else:
            print("Seperated previously")

'''
print('Number of fake images = {}'.format((len(os.listdir('dataset-dist/phase-01/training/fake'))-1)/2))
print('Number of pristine images = {}'.format((len(os.listdir('dataset-dist/phase-01/training/pristine/')))))

controller = "dataset-dist/phase-01/training/"
fraud_path = controller + "fake/"
clean_path = controller + "pristine/"
mask_path = controller + fraud_path + "masks/"

frauds = os.listdir(fraud_path)[1:] # contains both masked images as well as altered images one after the other
cleans = os.listdir(clean_path) # all prestine images

# seperate masked images from altered images
if not os.path.isdir(mask_path):
    os.mkdir(mask_path)
    for f in frauds:
        if len(f.split(".")) == 3:
            shutil.move(fraud_path + f, mask_path)
else:
    print("Seperated previously")
'''
'''
frauds = os.listdir(fraud_path)[1:-1]
fraud_image_shapes = []

# Collect fraudulent images and their 'shapes' aka image data
for f in frauds:
    fraud_image_shapes.append(imageio.imread(fraud_path + f).shape)

# Show image channels for fraudulent images
for i in len(fraud_image_shapes):
    print(str(i) + '\t' + str(fraud_image_shapes[i]) + '\t' + frauds[i])

'''
    




