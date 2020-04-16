from FraudulentImageCleaner import FraudulentImageCleaner
from ImageShapes import ImageShapes
import Model
from sklearn.model_selection import train_test_split
import os as os
import pickle
import shutil
import numpy as numpy
from tqdm import tqdm_notebook, tqdm
import pandas
from imageio import imread
import cv2
from multiprocessing import Manager, Process
import time

def count(mask):
    i = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c] == 255:
                i += 1
    return i
    
def samp_frauds(im, mask, stride, thres):
    kernal_size = 64
    samps = []

    for y in range(0, im.shape[0] - kernal_size + 1, stride):
        for x in range(0, im.shape[1] - kernal_size + 1, stride):
            count_ = count(mask[y:y + kernal_size, x:x + kernal_size])
            if (count_ > thres) and (kernal_size * kernal_size - count_ > thres):
                samps.append(im[y:y + kernal_size, x:x + kernal_size, :3])
    return samps

#print(len(x_train_fraud_images))
def process(batch, li, im, masks, stride, thres):
    for i, m in zip(im, masks):
        sam = samp_frauds(i, m, stride, thres)
        for s in sam:
            li.append(s)
        #print('Number of samples = ' + str(len(sam)))

def samp_rand(im, num, stride = 8):
    kernel_size = 64
    
    x = 0
    y = 0
    samples = []
    
    for y in range(0, im.shape[0] - kernel_size + 1, stride):
        for x in range(0, im.shape[1] - kernel_size + 1, stride):

            #c_255 = count_255(mask[y_start:y_start + kernel_size, x_start:x_start + kernel_size])

            #if (c_255 > threshold) and (kernel_size * kernel_size - c_255 > threshold):
            samples.append(im[y:y + kernel_size, x:x + kernel_size, :3])

    
    indices = numpy.random.randint(0, len(samples), min(len(samples), num))
    
    sampled = []
    for i in indices:
        sampled.append(samples[i])
    
    return sampled

def main():
    FIC = FraudulentImageCleaner()
    print('Number of fraud images and cleans ... ')
    print(FIC.get_num_frauds())
    print(FIC.get_num_clean())
    print('Setting fraud and clean images to list ...')
    FIC.set_frauds()
    FIC.set_cleans()
    print('Seperating masks from fraud images ... ')
    FIC.seperate_masks()
    print('Setting masked images to list ...')
    FIC.set_masks()

    masks = FIC.get_masks()
    cleans = FIC.get_cleans()
    frauds = FIC.get_frauds()
    fp = FIC.get_fraud_path()
    cp = FIC.get_clean_path()
    mp = FIC.get_mask_path()
    IS = ImageShapes(fp, mp, cp, frauds, masks, cleans)
    #print('Collecting shapes for fraudulent images ...')
    #IS.collect_shapes(show_f = True, show_m = False, show_c = False, verbose = True)
    #IS.set_hw(frauds, fp)
    #print('Max height of fraud images: {}'.format(IS.get_max_h()))
    #print('Min height of fraud images: {}'.format(IS.get_min_h()))
    #print('Max width of fraud images: {}'.format(IS.get_max_w()))
    #print('Min width of fraud images: {}'.format(IS.get_min_w()))
    #IS.dump()
    
    final_cleans = []
    for c in cleans:
        im = imread(cp + c)
        if len(im.shape) < 3 or im.shape[2] == 4:
            continue
        final_cleans.append(c)

    fraud_im = []
    final_frauds = []
    for f in frauds:
        im = imread(fp + f)
        try:
            fraud_im.append(im[:, :, :3])
            final_frauds.append(f)
        except IndexError:
            print(f'Image {f} only has one channel')

    images = []
    for i in range(len(final_cleans)):
        images.append(final_cleans[i])
    for i in range(len(final_frauds)):
        images.append(final_frauds[i])

    labels = [0] * 1025 + [1] * 450
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, stratify = labels)

    x_train_images = []
    for i in x_train:
        try:
            im = imread(cp + i)
        except FileNotFoundError:
            im = imread(fp + i)
        x_train_images.append(im)
    
    
    # Get the names for images that are masks
    x_train_mask_names = []
    for i, x in enumerate(x_train):
        if y_train[i] == 1:
            x_train_mask_names.append(x.split('.')[0] + '.mask.png')
    
    # Get fraud image names and the actual image
    x_train_fraud_names = []
    x_train_fraud_images = []
    for i, x in enumerate(x_train):
        if y_train[i] == 1:
            x_train_fraud_names.append(x)
            x_train_fraud_images.append(x_train_images[i])

    #with open('x_train_fraud_images.pickle', 'wb') as f:
        #pickle.dump(x_train_fraud_images, f)
    
    
    # Get clean image names and actual full image
    x_train_clean_names = []
    x_train_clean_images = []
    for i, x in enumerate(x_train):
        if y_train[i] == 0:
            x_train_clean_names.append(x)
            x_train_clean_images.append(x_train_images[i])
    
    
    # Get the actual masked images 
    x_train_masks = []
    for m in x_train_mask_names:
        im = imread(mp + m)
        if len(im.shape) > 2:
            im = im[:, :, 0]
        x_train_masks.append(im)

    #with open('x_train_masks.pickle', 'wb') as f:
        #pickle.dump(x_train_masks, f)
    
    '''samples_f = []
    i = 0
    for f, m in zip(x_train_fraud_images, x_train_masks):
        im_samp = samp_frauds(f, m)
        for s in im_samp:
            samples_f.append(s)
            i += 1'''

    #m = Manager()
    #c_li = m.list()
    '''
    print('Creating process for greyscale frauds ...')
    start_time = time.time()
    with open('x_train_masks.pickle', 'rb') as f:
        x_train_masks = pickle.load(f)

    with open('x_train_fraud_images.pickle', 'rb') as f:
        x_train_fraud_images = pickle.load(f)

    processes = []
    for batch in range(4):
        #process(batch, c_li, x_train_fraud_images[batch * 40:(batch + 1) * 40], x_train_masks[batch * 40:(batch + 1) * 40])
        processes.append(Process(target = process, args = (batch, c_li, x_train_fraud_images[batch * 40:(batch + 1) * 40], x_train_masks[batch * 40:(batch + 1) * 40], 32, 1600)))
    print('Starting process ...')
    for p in processes:
        p.start() 
    for p in processes:
        p.join()
    processes = []
    for batch in range(4, 9):
        #process(batch, c_li, x_train_fraud_images[batch * 40:(batch + 1) * 40], x_train_masks[batch * 40:(batch + 1) * 40])
        processes.append(Process(target = process, args = (batch, c_li, x_train_fraud_images[batch * 40:(batch + 1) * 40], x_train_masks[batch * 40:(batch + 1) * 40], 32, 1600)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    fraud_samples = c_li[0][numpy.newaxis, :, :, :]
    for i in c_li[1:]:
        fraud_samples = numpy.concatenate((fraud_samples, i[numpy.newaxis, :, :, :3]), axis = 0)

    print('Process finished ... Saving to "greyscalestride32/fraud_samples.npy" ...')
    numpy.save('greyscalestride32/fraud_samples.npy', fraud_samples)
    end_time = time.time()
    print('Elapsed time: {}'.format((end_time - start_time) / 60))

    '''
    '''
    print('Creating process for binary frauds ...')
    start_time = time.time()
    with open('x_train_masks.pickle', 'rb') as f:
        x_train_masks = pickle.load(f)

    with open('x_train_fraud_images.pickle', 'rb') as f:
        x_train_fraud_images = pickle.load(f)

    # Convert the greyscale (one-channel) images to binary
    bi = []
    for gs in x_train_masks:
        b = cv2.GaussianBlur(gs, (5,5), 0)
        ret, th = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bi.append(th)

    mask_pixels = [numpy.count_nonzero(~bi[i]) for i in range(len(bi))]

    processes = []
    for batch in range(4):
        #process(batch, c_li, x_train_fraud_images[batch * 40:(batch + 1) * 40], x_train_masks[batch * 40:(batch + 1) * 40])
        processes.append(Process(target = process, args = (batch, c_li, x_train_fraud_images[batch * 40:(batch + 1) * 40], bi[batch * 40:(batch + 1) * 40], 8, 1024)))
    print('Starting process ...')
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()
    
    processes = []
    for batch in range(4, 9):
        #process(batch, c_li, x_train_fraud_images[batch * 40:(batch + 1) * 40], x_train_masks[batch * 40:(batch + 1) * 40])
        processes.append(Process(target = process, args = (batch, c_li, x_train_fraud_images[batch * 40:(batch + 1) * 40], bi[batch * 40:(batch + 1) * 40], 8, 1024)))
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()
    
    fraud_samples = c_li[0][numpy.newaxis, :, :, :]
    for i in c_li[1:]:
        fraud_samples = numpy.concatenate((fraud_samples, i[numpy.newaxis, :, :, :3]), axis = 0)
    print('Process finished ... Saving to "binarystride8/fraud_samples.npy" ...')
    numpy.save('binarystride8/fraud_samples.npy', fraud_samples)
    end_time = time.time()
    print('Elapsed time: {}'.format((end_time - start_time) / 60))
    '''
    frauds_greyscale = numpy.load('greyscalestride32/fraud_samples.npy')
    frauds_binary = numpy.load('binarystride8/fraud_samples.npy')
    frauds_s = frauds_greyscale.shape[0]
    frauds_b = frauds_binary.shape[0]
    print(f'Number of ims {frauds_b}')
    print(f'Number of img {frauds_s}')
    print(len(x_train_clean_images))
    print(len(x_train_fraud_images))
    print(4893/820)
    print(6*820)
    print(179783/820)
    print(219*820)
    samples_clean_greyscale = numpy.ndarray(shape=(4920, 64, 64, 3), dtype=numpy.dtype('uint8'))
    i = 0
    for im in x_train_clean_images:
    
        samples = samp_rand(im, 6, stride=32)
        for sample in samples:
            samples_clean_greyscale[i, :, :, :] = sample
            i += 1
    print(i)

    samples_clean_binary = numpy.ndarray(shape=(179580, 64, 64, 3), dtype=numpy.dtype('uint8'))
    i = 0
    for im in x_train_clean_images:
    
        samples = samp_rand(im, 219)
        for s, sample in enumerate(samples):
            samples_clean_binary[i, :, :, :] = sample
            i += 1
    print(i)

    train_labels_binary = [0] * len(samples_clean_binary) + [1] * len(frauds_binary)
    train_labels_greyscale = [0] * len(samples_clean_greyscale) + [1] * len(frauds_greyscale)
    #numpy.save('binarystride8/clean_samples.npy', samples_clean_binary)
    #numpy.save('greyscalestride32/clean_samples.npy', samples_clean_greyscale)
    '''
    with open('x_train_images.pickle', 'wb') as f:
        pickle.dump(x_train_images, f)

    with open('x_train_clean_images.pickle', 'wb') as f:
        pickle.dump(x_train_clean_images, f)   

    with open('binarystride8/train_labels.pickle', 'wb') as f:
        pickle.dump(train_labels_binary, f) 
    with open('greyscalestride32/train_labels.pickle', 'wb') as f:
        pickle.dump(train_labels_binary, f)     
    with open('x_train_fraud_names.pickle', 'wb') as f:
        pickle.dump(x_train_fraud_names, f)
    with open('x_train_mask_names.pickle', 'wb') as f:
        pickle.dump(x_train_mask_names, f)
    with open('x_train_clean_names.pickle', 'wb') as f:
        pickle.dump(x_train_clean_names, f)
    '''
    train_data_greyscale = numpy.concatenate((samples_clean_greyscale, frauds_greyscale), axis=0)
    
    x_train, x_cv, y_train, y_cv = train_test_split(train_data_greyscale, train_labels_greyscale, test_size = 0.3, stratify = train_labels_greyscale)
    if not os.path.isdir('greyscalestride32/train_data'):
        os.mkdir('greyscalestride32/train_data')
        numpy.save('greyscalestride32/train_data/x_train.npy', x_train)
        numpy.save('greyscalestride32/train_data/x_cv.npy', x_cv)
        numpy.save('greyscalestride32/train_data/y_train.npy', y_train)
        numpy.save('greyscalestride32/train_data/y_cv.npy', y_cv)
    else:
        print('Path previously created ... training data already exists')

    train_data_binary = numpy.concatenate((samples_clean_binary, frauds_binary), axis=0)
    x_train, x_cv, y_train, y_cv = train_test_split(train_data_binary, train_labels_binary, test_size = 0.3, stratify = train_labels_binary)
    if not os.path.isdir('binarystride8/train_data'):
        os.mkdir('binarystride8/train_data')
        numpy.save('binarystride8/train_data/x_train.npy', x_train)
        numpy.save('binarystride8/train_data/x_cv.npy', x_cv)
        numpy.save('binarystride8/train_data/y_train.npy', y_train)
        numpy.save('binarystride8/train_data/y_cv.npy', y_cv)
    else:
        print('Path previously created ... taining data already exists')

if __name__ == '__main__':
    main()