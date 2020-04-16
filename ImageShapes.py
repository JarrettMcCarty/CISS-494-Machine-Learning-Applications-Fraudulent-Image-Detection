import os
import shutil
import numpy as numpy
import pickle
from imageio import imread
from tqdm import tqdm_notebook, tqdm
import pandas

class InvalidImageShapeException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None
    
    def __str__(self):
        if self.message:
            return 'InvalidImageShapeException: {0}'.format(self.message)
        else:
            return "InvalidImageShapeException raised\n"

class ImageShapes(object):
    ''' Used in conjuction with FraudulentImageCleaner '''
    

    def __init__(self, fp, mp, cp, f, m, c):
        if fp is None or fp == "": raise PathNotFoundException('No path provided for fraudulent images')
        if mp is None or mp == "": raise PathNotFoundException('No path provided for masked images')
        if cp is None or cp == "": raise PathNotFoundException('No path provided for clean images')
        if f is None or f is []: raise IndexError
        if m is None or m is []: raise IndexError
        if c is None or c is []: raise IndexError
        self.fraud_path = fp
        self.mask_path = mp
        self.clean_path = cp
        self.frauds = f
        self.masks = m
        self.cleans = c
        self.four_channel = []
        self.three_channel = []
        self.one_channel = []
        self.image_shapes = []
        self.heights = []
        self.widths = []

    def clear(self):
        del self.four_channel[:]
        del self.three_channel[:]
        del self.one_channel[:]
        del self.heights[:]
        del self.widths[:]

    def show_verbose(self, li, path, im_shapes):
        for i in range(len(im_shapes)):
            print(str(i) + '\t' + str(im_shapes[i]) + '\t' + li[i])

        for im in li:
            if imread(path + im).shape[2] > 4:
                print("More than 4 channels in: " + f)
            if imread(path + im).shape[2] < 3:
                print("Less than 3 channels in: " + f)
            if imread(path + im).shape[2] <= 2:
                print("2 or fewer channels in: " + f)

    ''' Image Shapes '''
    def collect_shapes(self, show_f = False, show_m = False, show_c = False, verbose = False):
        self.clear()
        if show_f:
            try:
                temp_frauds = os.listdir(self.fraud_path)[1:-1]
            except:
                raise PathNotFoundException('Invalid path when attempting to collect fraud shapes')
            for f in temp_frauds:
                self.image_shapes.append(imread(self.fraud_path + f).shape)
            if verbose:
                self.show_verbose(temp_frauds, self.fraud_path, self.image_shapes)

            for f in self.frauds:
                if imread(self.fraud_path + f).shape[2] == 4:
                    self.four_channel.append(f)
            self.three_channel = [f for f in self.frauds if f not in self.four_channel]
            return
        elif show_m:
            for m in self.masks:
                try:
                    self.image_shapes.append(imread(self.mask_path + m).shape)
                except:
                    raise PathNotFoundException('Invalid path when attempting to collect mask shapes')
                if verbose:
                    self.show_verbose(self.masks, self.mask_path, self.image_shapes)

                for m in self.masks:
                    if len(imread(self.mask_path + m).shape) == 2:
                        self.one_channel.append(m)
                    if len(imread(self.mask_path + m).shape) == 3 and imread(self.mask_path + m).shape[2] == 3:
                        self.three_channel.append(m)
                self.four_channel = [m for m in self.masks if ((m not in self.one_channel) and (m not in self.three_channel))]
                return
        elif show_c:
            for m in self.cleans:
                try:
                    self.image_shapes.append(imread(self.clean_path + m).shape)
                except:
                    raise PathNotFoundException('Invalid path when attempting to collect clean shapes')
                if verbose:
                    self.show_verbose(self.cleans, self.clean_path, self.image_shapes)

                for m in self.cleans:
                    if len(imread(self.clean_path + m).shape) < 3:
                        self.one_channel.append(m)
                    if len(imread(self.clean_path + m).shape) == 3 and imread(self.clean_path + m).shape[2] == 3:
                        self.three_channel.append(m)
                self.four_channel = [m for m in self.cleans if ((m not in self.one_channel) and (m not in self.three_channel))]
                return
        else:
            print("Nothing to collect")
            return

    def length_four_channel(self):
        return (len(self.four_channel))

    def length_three_channel(self):
        return (len(self.three_channel))

    def length_one_channel(self):
        return (len(self.one_channel)) # greyscale images

    def set_hw(self, li, path):
        for f in li:
            self.heights.append(imread(path + f).shape[0])
            self.widths.append(imread(path + f).shape[1])

    def get_max_h(self):
        return max(self.heights)

    def get_min_h(self):
        return min(self.heights)

    def get_max_w(self):
        return max(self.widths)

    def get_min_w(self):
        return min(self.widths)

    def compare(self):
        ''' Compares mask vs fraud shapes '''
        for m, f in tqdm_notebook(zip(m, f)):
            m_im = imread(self.mask_path + m)
            f_im = imread(self.fraud_path + f)
            if m_im.shape[:2] != f_im.shape[:2]:
                print(str(m_im.shape) + ' ' + str(f_im.shape))

    def table(self):
        t = {}
        self.collect_shapes(show_f = False, show_m = False, show_c = True, verbose = False)
        t['Cleans'] = [self.length_one_channel(), self.length_three_channel(), self.length_four_channel(), 1050]
        self.collect_shapes(show_f = True, show_m = False, show_c = False, verbose = False)
        t['Frauds'] = [self.length_one_channel(), self.length_three_channel(), self.length_four_channel(), 450]
        self.collect_shapes(show_f = False, show_m = True, show_c = False, verbose = False)
        t['Masks'] = [self.length_one_channel(), self.length_three_channel(), self.length_four_channel(), 450]
        pandas.DataFrame(t, index = ['One Channel', 'Three Channel', 'Four Channel', 'Totals'])

    def dump(self):
        self.collect_shapes(show_f = False, show_m = False, show_c = True, verbose = False)
        with open('OneChannelCleans.pickle', 'wb') as f:
            pickle.dump(self.one_channel, f)
        with open('ThreeChannelCleans.pickle', 'wb') as f:
            pickle.dump(self.three_channel, f)
        with open('FourChannelCleans.pickle', 'wb') as f:
            pickle.dump(self.four_channel, f)  

        self.collect_shapes(show_f = True, show_m = False, show_c = False, verbose = False)
        with open('ThreeChannelFrauds.pickle', 'wb') as f:
            pickle.dump(self.three_channel, f)
        with open('FourChannelFrauds.pickle', 'wb') as f:
            pickle.dump(self.four_channel, f)

        self.collect_shapes(show_f = False, show_m = True, show_c = False, verbose = False)
        with open('OneChannelMasks.pickle', 'wb') as f:
            pickle.dump(self.one_channel, f)
        with open('ThreeChannelMasks.pickle', 'wb') as f:
            pickle.dump(self.three_channel, f)
        with open('FourChannelMasks.pickle', 'wb') as f:
            pickle.dump(self.four_channel, f)

    def load(self, file):
        with open(file, 'rb') as f:
            ret = pickle.load(f)
        return ret
