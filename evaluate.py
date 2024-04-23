import matplotlib.pyplot as plt
from scipy.sparse.coo import coo_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import saveModel


def dice_iou(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    sum = np.sum(y_true_f) + np.sum(y_pred_f)
    smooth = 0.00001
    dice = (2. * intersection) / (sum + smooth)
    iou = dice / (2 - dice)
    return dice, iou

def evaluateModel(model, X_test, Y_test, batchSize, iters, results_save_path):

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)
    yp = np.round(yp,0)
    yp = yp.ravel().astype(int)

    Y_test = np.round(Y_test,0)
    Y_test = Y_test.ravel().astype(int)

    intersection = yp * Y_test
    union = yp + Y_test - intersection

    jaccard = (np.sum(intersection)/np.sum(union))  

    dice = (2. * jaccard ) / (jaccard + 1)


    print('Jaccard Index : '+str(jaccard))
    print('Dice Coefficient : '+str(dice))


    fp = open(results_save_path + '\\jaccard-{}.txt'.format(iters),'a')
    fp.write(str(jaccard)+'\n')
    fp.close()
    fp = open(results_save_path + '\\dice-{}.txt'.format(iters),'a')
    fp.write(str(dice)+'\n')
    fp.close()


    fp = open(results_save_path + '\\best-jaccard-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(jaccard>float(best)):
        print('***********************************************')
        print('Jaccard Index improved from '+str(best)+' to '+str(jaccard))
        fp = open(results_save_path + '\\best-jaccard-{}.txt'.format(iters),'w')
        fp.write(str(jaccard))
        fp.close()
        saveModel.saveModel(model, iters, results_save_path)


    fp = open(results_save_path + '\\best-dice-{}.txt'.format(iters),'r')
    best = fp.read()
    fp.close()
    if(dice>float(best)):
        print('***********************************************')
        print('Dice Index improved from '+str(best)+' to '+str(dice))
        fp = open(results_save_path + '\\best-dice-{}.txt'.format(iters),'w')
        fp.write(str(dice))
        fp.close()


