##################################################
############ AUTOR: JAROSLAV HOLAJ ###############
############### MATURITNÍ PRÁCE ##################
##################################################

##################################################

import math
import cv2
import numpy as np
import argparse
from skimage.metrics import structural_similarity
from BRISQUE_MODULES import *
import libsvm.svmutil as svmutil
from svm import *
from svmutil import *

##################################################

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--original", required=True)
ap.add_argument("-c", "--contrast", required=True)
args = vars(ap.parse_args())

# čtení obrazu z argumentů
imageA = cv2.imread(args["original"])
imageB = cv2.imread(args["contrast"])
# převedení do odstínů šedi
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

#####################################################

def PSNR(img1, img2):
    #
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    #
    PIXEL_MAX = 255.0
    #
    PSNRScore = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNRScore

######################################################

def SSIM(img1, img2):
    (SSIMScore, diff) = structural_similarity(img1, img2, full=True)
    diff = (diff * 255).astype("uint8")
    return SSIMScore

######################################################

def BRISQUE(img):
    # read image from given path

    # compute feature vectors of the image
    features = compute_features(grayA)

    # rescale the brisqueFeatures vector from -1 to 1
    x = [0]
    
    # pre loaded lists from C++ Module to rescale brisquefeatures vector to [-1, 1]
    min_= [0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351]
    
    max_= [9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484]

    # append the rescaled vector to x 
    for i in range(0, 36):
        min = min_[i]
        max = max_[i] 
        x.append(-1 + (2.0/(max - min) * (features[i] - min)))
    
    # load model 
    model = svmutil.svm_load_model("allmodel")

    # create svm node array from python list
    x, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
    x[36].index = -1 # set last index to -1 to indicate the end.
	
	# get important parameters from model
    svm_type = model.get_svm_type()
    is_prob_model = model.is_probability_model()
    nr_class = model.get_nr_class()
    
    if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
        # here svm_type is EPSILON_SVR as it's regression problem
        nr_classifier = 1
    dec_values = (c_double * nr_classifier)()
    
    # calculate the quality score of the image using the model and svm_node_array
    qualityscore = svmutil.libsvm.svm_predict_probability(model, x, dec_values)

    return qualityscore
 ###########################################################################

brisqueOut = round(BRISQUE(imageA), 2)
ssimOut = round(SSIM(grayA,grayB)*100, 2)
psnrOut = round(PSNR(imageA,imageB), 2)
qualList = [brisqueOut,ssimOut]
avgQual = round(sum(qualList)/len(qualList),2)

print("-------------------------------------------")
print("BRISQUE KVALITA: {}".format(brisqueOut))
print("SSIM KVALITA: {}".format(ssimOut))
print("PSNR KVALITA: {}".format(psnrOut))
print("-------------------SOUHRN------------------")
print("ZPRŮMĚROVANÁ KVALITA (BRISQUE, SSIM): {}".format(avgQual))
print("-------------------------------------------")

