from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.optimizers import *
from tensorflow.keras.backend import clear_session
import os
import numpy as np
import time
from tqdm import *
import cv2
from spectral import *
import TrainStep
import models
from tensorflow import keras
from scipy.spatial.distance import directed_hausdorff,cdist
import scipy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def calculate_hausdorff_distance(seg_image1, seg_image2):
    segmentation_boundary = scipy.ndimage.morphology.binary_dilation(seg_image1 ) - seg_image1 
    ground_truth_boundary = scipy.ndimage.morphology.binary_dilation(seg_image2 ) - seg_image2 

    coords1 = np.argwhere(segmentation_boundary)
    coords2 = np.argwhere(ground_truth_boundary)

    if len(coords1) == 0 or len(coords2) == 0:
        return 0

    distances1 = cdist(coords1, coords2, 'euclidean')
    distances2 = cdist(coords2, coords1, 'euclidean')

    hausdorff_distance1 = np.max(np.min(distances1, axis=1))
    hausdorff_distance2 = np.max(np.min(distances2, axis=1))

    return max(hausdorff_distance1, hausdorff_distance2)

def calculate_assd(seg_image1, seg_image2):
    segmentation_boundary = scipy.ndimage.morphology.binary_dilation(seg_image1 ) - seg_image1 
    ground_truth_boundary = scipy.ndimage.morphology.binary_dilation(seg_image2 ) - seg_image2 

    coords1 = np.argwhere(segmentation_boundary)
    coords2 = np.argwhere(ground_truth_boundary)

    if len(coords1) == 0 or len(coords2) == 0:
        return 0
    
    distances1 = cdist(coords1, coords2, 'euclidean')
    distances2 = cdist(coords2, coords1, 'euclidean')

    assd1 = np.min(distances1, axis=1)
    assd2 = np.min(distances2, axis=1)

    assd = (np.sum(assd1) + np.sum(assd2)) / (np.sum(segmentation_boundary ) + np.sum(ground_truth_boundary))

    return assd

def cal_hd_assd(seg_image1, seg_image2):
    segmentation_boundary = scipy.ndimage.morphology.binary_dilation(seg_image1 ) - seg_image1 
    ground_truth_boundary = scipy.ndimage.morphology.binary_dilation(seg_image2 ) - seg_image2 

    coords1 = np.argwhere(segmentation_boundary)
    coords2 = np.argwhere(ground_truth_boundary)

    if len(coords1) == 0 or len(coords2) == 0:
        return 0,0

    distances1 = cdist(coords1, coords2, 'euclidean')
    distances2 = cdist(coords2, coords1, 'euclidean')

    hausdorff_distance1 = np.max(np.min(distances1, axis=1))
    hausdorff_distance2 = np.max(np.min(distances2, axis=1))

    hausdorff_distance = max(hausdorff_distance1, hausdorff_distance2)

    assd1 = np.min(distances1, axis=1)
    assd2 = np.min(distances2, axis=1)

    assd = (np.sum(assd1) + np.sum(assd2)) / (np.sum(segmentation_boundary ) + np.sum(ground_truth_boundary))

    return hausdorff_distance, assd

def BCE():
    def dice(y_true, y_pred):
        return tf.keras.metrics.binary_crossentropy(y_true, y_pred)
    return dice

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0:1], device_type='GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# img_path = '..\\Input'
# msk_path = '..\\GroundTruth'
# tst_path = '..\\test'

img_path = 'E:\\ProjectFiles\\Data\\CVC_ClinicDB\\images'
msk_path = 'E:\\ProjectFiles\\Data\\CVC_ClinicDB\\masks'
tst_path = 'E:\\ProjectFiles\\Data\\CVC_ClinicDB\\test'

model_results_dir = 'E:\\ProjectFiles\\models\\CVC_ClinicDB\\2024'
if not os.path.exists(model_results_dir):
    os.makedirs(model_results_dir)

model_name = 'unet'
dataset = 'ChoRGB' 
pretrained_weights = None



img_resize = (256,192)
input_size = (192,256,3)



epochs = 150 
batchSize = 8
learning_rate = 0.001


img_files = next(os.walk(img_path))[2]
msk_files = next(os.walk(msk_path))[2]
tst_files = next(os.walk(tst_path))[2]

img_files.sort() 
msk_files.sort()
tst_files.sort()

X = []
Y = []
save_name = []

for img_fl in tqdm(msk_files): 
    image = cv2.imread(img_path + '\\' + img_fl, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(image, img_resize)
    X.append(resized_img)

    img_name_img = img_fl.split('.')[0]
    save_name.append(img_name_img)

    mask = cv2.imread(msk_path + '\\' + img_fl, cv2.IMREAD_GRAYSCALE)
    resized_msk = cv2.resize(mask, img_resize)
    Y.append(resized_msk)

X = np.array(X)
Y = np.array(Y)
save_name = np.array(save_name)

# X = X.reshape((X.shape[0],X.shape[1],X.shape[2],1))  # Grayscale input images
Y = Y.reshape((Y.shape[0],Y.shape[1],Y.shape[2],1))

np.random.seed(1000)
shuffle_indices = np.random.permutation(np.arange(len(Y)))
x_shuffled = X[shuffle_indices]
y_shuffled = Y[shuffle_indices]
save_name_shuffled = save_name[shuffle_indices]

x_shuffled = x_shuffled / 255
y_shuffled = y_shuffled / 255
y_shuffled = np.round(y_shuffled,0)

print(x_shuffled.shape)
print(y_shuffled.shape)

length = int(float(len(x_shuffled))/5)

for i in range(0,5):
    tic = time.ctime()
    fp = open(model_results_dir +'\\jaccard-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()
    fp = open(model_results_dir +'\\dice-{}.txt'.format(i),'w')
    fp.write(str(tic) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-jaccard-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()
    fp = open(model_results_dir +'\\best-dice-{}.txt'.format(i),'w')
    fp.write('-1.0')
    fp.close()

    index = int(float(len(x_shuffled))*(i+1)/5)
    x_train = np.concatenate((x_shuffled[:index-length], x_shuffled[index:]), axis=0)
    x_val = x_shuffled[index-length:index]
    y_train = np.concatenate((y_shuffled[:index-length], y_shuffled[index:]), axis=0)
    y_val = y_shuffled[index-length:index]
    save_name_img = save_name_shuffled[index-length:index]


    model = models.MAEFNet(pretrained_weights = None,input_size = input_size) 
    model.summary()
    print ('iter: %s' % (str(i)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    TrainStep.trainStep(model, x_train, y_train, x_val, y_val, epochs=epochs, batchSize=batchSize, iters = i, results_save_path = model_results_dir)


    fp = open(model_results_dir +'\\best-jaccard-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-jaccard.txt','a')
    tic = time.ctime()
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()

    fp = open(model_results_dir +'\\best-dice-{}.txt'.format(i),'r')
    best = fp.read()
    print(best)
    fp.close()
    fp = open(model_results_dir +'\\epoch_best-dice.txt','a')
    fp.write('iter: ' + str(i) + '\n' + str(tic) + ':   ' + str(best) + '\n')
    fp.close()
    
    clear_session()
    tf.compat.v1.reset_default_graph()

