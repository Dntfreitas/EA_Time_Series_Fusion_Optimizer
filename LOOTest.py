"""
Code to run LOO with the solutions found by the GA and PSO
"""
"""
Libraries
"""
import gc
import pickle

import numpy as np
import scipy.io as spio
import tensorflow as tf
from keras.layers import Dense, Dropout, Input, concatenate, Bidirectional, LSTM
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
Control variables
"""
patience = 10  # number of consecutive epochs without performance improvement to run the training
BatchSize = 1024  # size of the minibatches
features = 100  # number of points to be considered fo each time step of the LSTM
startEpochs = 0  # nummber of the subejct to start the analysis
Epochs = 16  # number of subjects to be examined
BeginTest = 15  # location in the subjetcs information array which specifies wich subject should be used to create the test dataset
EpochsWork = 1  # number of iterations for each LOO cycle
Begin = 0  # number of the subject to start the LOO train dataset
numberSubjects = 16  # number of examined subjetcs
# chromosome = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1] # FOR GA
chromosome = [1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]  # FOR PSO
# os.chdir('D:\MatlabCode\PytonTests') # location of the dataset
"""
Decode the chromosome/particle
"""
if chromosome[0] == False:
    if chromosome[1] == False:
        if chromosome[2] == False:
            NumberOfCannels = 1
        else:
            NumberOfCannels = 2
    else:
        if chromosome[2] == False:
            NumberOfCannels = 3
        else:
            NumberOfCannels = 4
else:
    if chromosome[1] == False:
        if chromosome[2] == False:
            NumberOfCannels = 5
        else:
            NumberOfCannels = 6
    else:
        NumberOfCannels = 7
if chromosome[3] == False:
    if chromosome[4] == False:
        NumberOfTimeSteps = 10
    else:
        NumberOfTimeSteps = 15
else:
    if chromosome[4] == False:
        NumberOfTimeSteps = 20
    else:
        NumberOfTimeSteps = 25
if chromosome[5] == False:
    NumberOfLSTMLayers = 1
else:
    NumberOfLSTMLayers = 2
if chromosome[6] == False:
    TypeLSTMLayer = 0
else:
    TypeLSTMLayer = 1
if chromosome[7] == False:
    if chromosome[8] == False:
        ShapeOfLSTM = 100
    else:
        ShapeOfLSTM = 200
else:
    if chromosome[8] == False:
        ShapeOfLSTM = 300
    else:
        ShapeOfLSTM = 400
if chromosome[9] == False:
    if chromosome[10] == False:
        PercentageDropout = 0
    else:
        PercentageDropout = 0.05
else:
    if chromosome[10] == False:
        PercentageDropout = 0.1
    else:
        PercentageDropout = 0.15
if chromosome[11] == False:
    if chromosome[12] == False:
        ShapeOfDenseLayer = 0
    else:
        ShapeOfDenseLayer = 200
else:
    if chromosome[12] == False:
        ShapeOfDenseLayer = 300
    else:
        ShapeOfDenseLayer = 400
if chromosome[13] == False:
    if chromosome[14] == False:
        ActivationFunction = 0  # tanh
    else:
        ActivationFunction = 1  # sigmoid
else:
    if chromosome[14] == False:
        ActivationFunction = 2  # relu
    else:
        ActivationFunction = 3  # selu
"""
Variables to store the results of the LOO cycles
"""
AccLOO = np.zeros(Epochs)
SenLOO = np.zeros(Epochs)
SpeLOO = np.zeros(Epochs)
AUCLOO = np.zeros(Epochs)
TPLOO = np.zeros(Epochs)
TNLOO = np.zeros(Epochs)
FPLOO = np.zeros(Epochs)
FNLOO = np.zeros(Epochs)
"""
LOO cycles
"""
for ee in range(startEpochs, Epochs, 1):
    print('\n\n Subject: ', ee)
    """
    Variables to store the results of the interations for each of the LOO cycles
    """
    AccAtEnd = np.zeros(EpochsWork)
    SenAtEnd = np.zeros(EpochsWork)
    SpeAtEnd = np.zeros(EpochsWork)
    AUCAtEnd = np.zeros(EpochsWork)
    TPEnd = np.zeros(EpochsWork)
    TNEnd = np.zeros(EpochsWork)
    FPEnd = np.zeros(EpochsWork)
    FNEnd = np.zeros(EpochsWork)
    """
    Interations for each of the LOO cycles
    """
    for ff in range(EpochsWork):
        print('\n\n Cycle: ', ee)
        """
        Load the dataset
        """
        mat = spio.loadmat('n1F2V2.mat', squeeze_me=True)
        n1F2 = mat.get('x')
        del mat
        mat = spio.loadmat('n2F2V2.mat', squeeze_me=True)
        n2F2 = mat.get('x')
        del mat
        mat = spio.loadmat('n3F2V2.mat', squeeze_me=True)
        n3F2 = mat.get('x')
        del mat
        mat = spio.loadmat('n4F2V2.mat', squeeze_me=True)
        n4F2 = mat.get('x')
        del mat
        mat = spio.loadmat('n5F2V2.mat', squeeze_me=True)
        n5F2 = mat.get('x')
        del mat
        mat = spio.loadmat('n10F2V2.mat', squeeze_me=True)
        n10F2 = mat.get('x')
        del mat
        mat = spio.loadmat('n11F2V2.mat', squeeze_me=True)
        n11F2 = mat.get('x')
        del mat
        mat = spio.loadmat('n16F2V2.mat', squeeze_me=True)
        n16F2 = mat.get('x')
        del mat
        mat = spio.loadmat('nfle1F2V2.mat', squeeze_me=True)
        nfle1F2 = mat.get('f2')
        del mat
        mat = spio.loadmat('nfle2F2V2.mat', squeeze_me=True)
        nfle2F2 = mat.get('f2')
        del mat
        mat = spio.loadmat('nfle4F2V2.mat', squeeze_me=True)
        nfle4F2 = mat.get('f2')
        del mat
        mat = spio.loadmat('nfle7F2V2.mat', squeeze_me=True)
        nfle7F2 = mat.get('f2')
        del mat
        mat = spio.loadmat('nfle12F2V2.mat', squeeze_me=True)
        nfle12F2 = mat.get('f2')
        del mat
        mat = spio.loadmat('nfle13F2V2.mat', squeeze_me=True)
        nfle13F2 = mat.get('f2')
        del mat
        mat = spio.loadmat('nfle14F2V2.mat', squeeze_me=True)
        nfle14F2 = mat.get('f2')
        del mat
        mat = spio.loadmat('nfle15F2V2.mat', squeeze_me=True)
        nfle15F2 = mat.get('f2')
        del mat
        mat = spio.loadmat('n1C4V2.mat', squeeze_me=True)
        n1C4 = mat.get('x')
        del mat
        mat = spio.loadmat('n2C4V2.mat', squeeze_me=True)
        n2C4 = mat.get('x')
        del mat
        mat = spio.loadmat('n3C4V2.mat', squeeze_me=True)
        n3C4 = mat.get('x')
        del mat
        mat = spio.loadmat('n4C4V2.mat', squeeze_me=True)
        n4C4 = mat.get('x')
        del mat
        mat = spio.loadmat('n5C4V2.mat', squeeze_me=True)
        n5C4 = mat.get('x')
        del mat
        mat = spio.loadmat('n10C4V2.mat', squeeze_me=True)
        n10C4 = mat.get('x')
        del mat
        mat = spio.loadmat('n11C4V2.mat', squeeze_me=True)
        n11C4 = mat.get('x')
        del mat
        mat = spio.loadmat('n16C4V2.mat', squeeze_me=True)
        n16C4 = mat.get('x')
        del mat
        mat = spio.loadmat('nfle1C4V2.mat', squeeze_me=True)
        nfle1C4 = mat.get('c4')
        del mat
        mat = spio.loadmat('nfle2C4V2.mat', squeeze_me=True)
        nfle2C4 = mat.get('c4')
        del mat
        mat = spio.loadmat('nfle4C4V2.mat', squeeze_me=True)
        nfle4C4 = mat.get('c4')
        del mat
        mat = spio.loadmat('nfle7C4V2.mat', squeeze_me=True)
        nfle7C4 = mat.get('c4')
        del mat
        mat = spio.loadmat('nfle12C4V2.mat', squeeze_me=True)
        nfle12C4 = mat.get('c4')
        del mat
        mat = spio.loadmat('nfle13C4V2.mat', squeeze_me=True)
        nfle13C4 = mat.get('c4')
        del mat
        mat = spio.loadmat('nfle14C4V2.mat', squeeze_me=True)
        nfle14C4 = mat.get('c4')
        del mat
        mat = spio.loadmat('nfle15C4V2.mat', squeeze_me=True)
        nfle15C4 = mat.get('c4')
        del mat
        mat = spio.loadmat('n1F4V2.mat', squeeze_me=True)
        n1F4 = mat.get('x')
        del mat
        mat = spio.loadmat('n2F4V2.mat', squeeze_me=True)
        n2F4 = mat.get('x')
        del mat
        mat = spio.loadmat('n3F4V2.mat', squeeze_me=True)
        n3F4 = mat.get('x')
        del mat
        mat = spio.loadmat('n4F4V2.mat', squeeze_me=True)
        n4F4 = mat.get('x')
        del mat
        mat = spio.loadmat('n5F4V2.mat', squeeze_me=True)
        n5F4 = mat.get('x')
        del mat
        mat = spio.loadmat('n10F4V2.mat', squeeze_me=True)
        n10F4 = mat.get('x')
        del mat
        mat = spio.loadmat('n11F4V2.mat', squeeze_me=True)
        n11F4 = mat.get('x')
        del mat
        mat = spio.loadmat('n16F4V2.mat', squeeze_me=True)
        n16F4 = mat.get('x')
        del mat
        mat = spio.loadmat('nfle1F4V2.mat', squeeze_me=True)
        nfle1F4 = mat.get('f4')
        del mat
        mat = spio.loadmat('nfle2F4V2.mat', squeeze_me=True)
        nfle2F4 = mat.get('f4')
        del mat
        mat = spio.loadmat('nfle4F4V2.mat', squeeze_me=True)
        nfle4F4 = mat.get('f4')
        del mat
        mat = spio.loadmat('nfle7F4V2.mat', squeeze_me=True)
        nfle7F4 = mat.get('f4')
        del mat
        mat = spio.loadmat('nfle12F4V2.mat', squeeze_me=True)
        nfle12F4 = mat.get('f4')
        del mat
        mat = spio.loadmat('nfle13F4V2.mat', squeeze_me=True)
        nfle13F4 = mat.get('f4')
        del mat
        mat = spio.loadmat('nfle14F4V2.mat', squeeze_me=True)
        nfle14F4 = mat.get('f4')
        del mat
        mat = spio.loadmat('nfle15F4V2.mat', squeeze_me=True)
        nfle15F4 = mat.get('f4')
        del mat
        mat = spio.loadmat('n1eegminutLable2.mat', squeeze_me=True)
        nc1 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('n2eegminutLable2.mat', squeeze_me=True)
        nc2 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('n3eegminutLable2.mat', squeeze_me=True)
        nc3 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('n4eegminutLable2.mat', squeeze_me=True)
        nc4 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('n5eegminutLable2.mat', squeeze_me=True)
        nc5 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('n10eegminutLable2.mat', squeeze_me=True)
        nc10 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('n11eegminutLable2.mat', squeeze_me=True)
        nc11 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('n16eegminutLable2.mat', squeeze_me=True)
        nc16 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('nfle1eegminutLable2.mat', squeeze_me=True)
        nflec1 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('nfle2eegminutLable2.mat', squeeze_me=True)
        nflec2 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('nfle4eegminutLable2.mat', squeeze_me=True)
        nflec4 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('nfle7eegminutLable2.mat', squeeze_me=True)
        nflec7 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('nfle12eegminutLable2.mat', squeeze_me=True)
        nflec12 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('nfle13eegminutLable2.mat', squeeze_me=True)
        nflec13 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('nfle14eegminutLable2.mat', squeeze_me=True)
        nflec14 = mat.get('CAPlabel1')
        del mat
        mat = spio.loadmat('nfle15eegminutLable2.mat', squeeze_me=True)
        nflec15 = mat.get('CAPlabel1')
        del mat
        """
        Standardize the dataset
        """
        n1F2 = (n1F2 - np.mean(n1F2)) / np.std(n1F2)
        n2F2 = (n2F2 - np.mean(n2F2)) / np.std(n2F2)
        n3F2 = (n3F2 - np.mean(n3F2)) / np.std(n3F2)
        n4F2 = (n4F2 - np.mean(n4F2)) / np.std(n4F2)
        n5F2 = (n5F2 - np.mean(n5F2)) / np.std(n5F2)
        n10F2 = (n10F2 - np.mean(n10F2)) / np.std(n10F2)
        n11F2 = (n11F2 - np.mean(n11F2)) / np.std(n11F2)
        n16F2 = (n16F2 - np.mean(n16F2)) / np.std(n16F2)
        nfle1F2 = (nfle1F2 - np.mean(nfle1F2)) / np.std(nfle1F2)
        nfle2F2 = (nfle2F2 - np.mean(nfle2F2)) / np.std(nfle2F2)
        nfle4F2 = (nfle4F2 - np.mean(nfle4F2)) / np.std(nfle4F2)
        nfle7F2 = (nfle7F2 - np.mean(nfle7F2)) / np.std(nfle7F2)
        nfle12F2 = (nfle12F2 - np.mean(nfle12F2)) / np.std(nfle12F2)
        nfle13F2 = (nfle13F2 - np.mean(nfle13F2)) / np.std(nfle13F2)
        nfle14F2 = (nfle14F2 - np.mean(nfle14F2)) / np.std(nfle14F2)
        nfle15F2 = (nfle15F2 - np.mean(nfle15F2)) / np.std(nfle15F2)
        n1C4 = (n1C4 - np.mean(n1C4)) / np.std(n1C4)
        n2C4 = (n2C4 - np.mean(n2C4)) / np.std(n2C4)
        n3C4 = (n3C4 - np.mean(n3C4)) / np.std(n3C4)
        n4C4 = (n4C4 - np.mean(n4C4)) / np.std(n4C4)
        n5C4 = (n5C4 - np.mean(n5C4)) / np.std(n5C4)
        n10C4 = (n10C4 - np.mean(n10C4)) / np.std(n10C4)
        n11C4 = (n11C4 - np.mean(n11C4)) / np.std(n11C4)
        n16C4 = (n16C4 - np.mean(n16C4)) / np.std(n16C4)
        nfle1C4 = (nfle1C4 - np.mean(nfle1C4)) / np.std(nfle1C4)
        nfle2C4 = (nfle2C4 - np.mean(nfle2C4)) / np.std(nfle2C4)
        nfle4C4 = (nfle4C4 - np.mean(nfle4C4)) / np.std(nfle4C4)
        nfle7C4 = (nfle7C4 - np.mean(nfle7C4)) / np.std(nfle7C4)
        nfle12C4 = (nfle12C4 - np.mean(nfle12C4)) / np.std(nfle12C4)
        nfle13C4 = (nfle13C4 - np.mean(nfle13C4)) / np.std(nfle13C4)
        nfle14C4 = (nfle14C4 - np.mean(nfle14C4)) / np.std(nfle14C4)
        nfle15C4 = (nfle15C4 - np.mean(nfle15C4)) / np.std(nfle15C4)
        n1F4 = (n1F4 - np.mean(n1F4)) / np.std(n1F4)
        n2F4 = (n2F4 - np.mean(n2F4)) / np.std(n2F4)
        n3F4 = (n3F4 - np.mean(n3F4)) / np.std(n3F4)
        n4F4 = (n4F4 - np.mean(n4F4)) / np.std(n4F4)
        n5F4 = (n5F4 - np.mean(n5F4)) / np.std(n5F4)
        n10F4 = (n10F4 - np.mean(n10F4)) / np.std(n10F4)
        n11F4 = (n11F4 - np.mean(n11F4)) / np.std(n11F4)
        n16F4 = (n16F4 - np.mean(n16F4)) / np.std(n16F4)
        nfle1F4 = (nfle1F4 - np.mean(nfle1F4)) / np.std(nfle1F4)
        nfle2F4 = (nfle2F4 - np.mean(nfle2F4)) / np.std(nfle2F4)
        nfle4F4 = (nfle4F4 - np.mean(nfle4F4)) / np.std(nfle4F4)
        nfle7F4 = (nfle7F4 - np.mean(nfle7F4)) / np.std(nfle7F4)
        nfle12F4 = (nfle12F4 - np.mean(nfle12F4)) / np.std(nfle12F4)
        nfle13F4 = (nfle13F4 - np.mean(nfle13F4)) / np.std(nfle13F4)
        nfle14F4 = (nfle14F4 - np.mean(nfle14F4)) / np.std(nfle14F4)
        nfle15F4 = (nfle15F4 - np.mean(nfle15F4)) / np.std(nfle15F4)
        """
        Creat the array to specify the subjects to be used for the train and test dataset
        """
        if ee == 0:
            normalSubjects = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0])
        elif ee == 1:
            normalSubjects = np.array([0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1])
        elif ee == 2:
            normalSubjects = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 2])
        elif ee == 3:
            normalSubjects = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3])
        elif ee == 4:
            normalSubjects = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 4])
        elif ee == 5:
            normalSubjects = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 5])
        elif ee == 6:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 6])
        elif ee == 7:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 7])
        elif ee == 8:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 8])
        elif ee == 9:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 9])
        elif ee == 10:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 10])
        elif ee == 11:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 11])
        elif ee == 12:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 12])
        elif ee == 13:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 13])
        elif ee == 14:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14])
        else:
            normalSubjects = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        YTrain = []
        YTest = []
        """
        Creat the train and test dataset
        """
        if normalSubjects[Begin] == 0:
            XTrainF2 = n1F2
            XTrainC4 = n1C4
            XTrainF4 = n1F4
            YTrain = nc1
        elif normalSubjects[Begin] == 1:
            XTrainF2 = n2F2
            XTrainC4 = n2C4
            XTrainF4 = n2F4
            YTrain = nc2
        elif normalSubjects[Begin] == 2:
            XTrainF2 = n3F2
            XTrainC4 = n3C4
            XTrainF4 = n3F4
            YTrain = nc3
        elif normalSubjects[Begin] == 3:
            XTrainF2 = n4F2
            XTrainC4 = n4C4
            XTrainF4 = n4F4
            YTrain = nc4
        elif normalSubjects[Begin] == 4:
            XTrainF2 = n5F2
            XTrainC4 = n5C4
            XTrainF4 = n5F4
            YTrain = nc5
        elif normalSubjects[Begin] == 5:
            XTrainF2 = n10F2
            XTrainC4 = n10C4
            XTrainF4 = n10F4
            YTrain = nc10
        elif normalSubjects[Begin] == 6:
            XTrainF2 = n11F2
            XTrainC4 = n11C4
            XTrainF4 = n11F4
            YTrain = nc11
        elif normalSubjects[Begin] == 7:
            XTrainF2 = n16F2
            XTrainC4 = n16C4
            XTrainF4 = n16F4
            YTrain = nc16
        elif normalSubjects[Begin] == 8:
            XTrainF2 = nfle1F2
            XTrainC4 = nfle1C4
            XTrainF4 = nfle1F4
            YTrain = nflec1
        elif normalSubjects[Begin] == 9:
            XTrainF2 = nfle2F2
            XTrainC4 = nfle2C4
            XTrainF4 = nfle2F4
            YTrain = nflec2
        elif normalSubjects[Begin] == 10:
            XTrainF2 = nfle4F2
            XTrainC4 = nfle4C4
            XTrainF4 = nfle4F4
            YTrain = nflec4
        elif normalSubjects[Begin] == 11:
            XTrainF2 = nfle7F2
            XTrainC4 = nfle7C4
            XTrainF4 = nfle7F4
            YTrain = nflec7
        elif normalSubjects[Begin] == 12:
            XTrainF2 = nfle12F2
            XTrainC4 = nfle12C4
            XTrainF4 = nfle12F4
            YTrain = nflec12
        elif normalSubjects[Begin] == 13:
            XTrainF2 = nfle13F2
            XTrainC4 = nfle13C4
            XTrainF4 = nfle13F4
            YTrain = nflec13
        elif normalSubjects[Begin] == 14:
            XTrainF2 = nfle14F2
            XTrainC4 = nfle14C4
            XTrainF4 = nfle14F4
            YTrain = nflec14
        else:
            XTrainF2 = nfle15F2
            XTrainC4 = nfle15C4
            XTrainF4 = nfle15F4
            YTrain = nflec15
        if normalSubjects[BeginTest] == 0:
            XTestF2 = n1F2
            XTestC4 = n1C4
            XTestF4 = n1F4
            YTest = nc1
        elif normalSubjects[BeginTest] == 1:
            XTestF2 = n2F2
            XTestC4 = n2C4
            XTestF4 = n2F4
            YTest = nc2
        elif normalSubjects[BeginTest] == 2:
            XTestF2 = n3F2
            XTestC4 = n3C4
            XTestF4 = n3F4
            YTest = nc3
        elif normalSubjects[BeginTest] == 3:
            XTestF2 = n4F2
            XTestC4 = n4C4
            XTestF4 = n4F4
            YTest = nc4
        elif normalSubjects[BeginTest] == 4:
            XTestF2 = n5F2
            XTestC4 = n5C4
            XTestF4 = n5F4
            YTest = nc5
        elif normalSubjects[BeginTest] == 5:
            XTestF2 = n10F2
            XTestC4 = n10C4
            XTestF4 = n10F4
            YTest = nc10
        elif normalSubjects[BeginTest] == 6:
            XTestF2 = n11F2
            XTestC4 = n11C4
            XTestF4 = n11F4
            YTest = nc11
        elif normalSubjects[BeginTest] == 7:
            XTestF2 = n16F2
            XTestC4 = n16C4
            XTestF4 = n16F4
            YTest = nc16
        elif normalSubjects[BeginTest] == 8:
            XTestF2 = nfle1F2
            XTestC4 = nfle1C4
            XTestF4 = nfle1F4
            YTest = nflec1
        elif normalSubjects[BeginTest] == 9:
            XTestF2 = nfle2F2
            XTestC4 = nfle2C4
            XTestF4 = nfle2F4
            YTest = nflec2
        elif normalSubjects[BeginTest] == 10:
            XTestF2 = nfle4F2
            XTestC4 = nfle4C4
            XTestF4 = nfle4F4
            YTest = nflec4
        elif normalSubjects[BeginTest] == 11:
            XTestF2 = nfle7F2
            XTestC4 = nfle7C4
            XTestF4 = nfle7F4
            YTest = nflec7
        elif normalSubjects[BeginTest] == 12:
            XTestF2 = nfle12F2
            XTestC4 = nfle12C4
            XTestF4 = nfle12F4
            YTest = nflec12
        elif normalSubjects[BeginTest] == 13:
            XTestF2 = nfle13F2
            XTestC4 = nfle13C4
            XTestF4 = nfle13F4
            YTest = nflec13
        elif normalSubjects[BeginTest] == 14:
            XTestF2 = nfle14F2
            XTestC4 = nfle14C4
            XTestF4 = nfle14F4
            YTest = nflec14
        else:
            XTestF2 = nfle15F2
            XTestC4 = nfle15C4
            XTestF4 = nfle15F4
            YTest = nflec15
        for x in range(20):
            if x < BeginTest and x > Begin:
                if normalSubjects[x] == 0:
                    XTrainF2 = np.concatenate((XTrainF2, n1F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, n1C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, n1F4), axis=0)
                    YTrain = np.concatenate((YTrain, nc1), axis=0)
                if normalSubjects[x] == 1:
                    XTrainF2 = np.concatenate((XTrainF2, n2F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, n2C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, n2F4), axis=0)
                    YTrain = np.concatenate((YTrain, nc2), axis=0)
                if normalSubjects[x] == 2:
                    XTrainF2 = np.concatenate((XTrainF2, n3F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, n3C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, n3F4), axis=0)
                    YTrain = np.concatenate((YTrain, nc3), axis=0)
                if normalSubjects[x] == 3:
                    XTrainF2 = np.concatenate((XTrainF2, n4F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, n4C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, n4F4), axis=0)
                    YTrain = np.concatenate((YTrain, nc4), axis=0)
                if normalSubjects[x] == 4:
                    XTrainF2 = np.concatenate((XTrainF2, n5F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, n5C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, n5F4), axis=0)
                    YTrain = np.concatenate((YTrain, nc5), axis=0)
                if normalSubjects[x] == 5:
                    XTrainF2 = np.concatenate((XTrainF2, n10F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, n10C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, n10F4), axis=0)
                    YTrain = np.concatenate((YTrain, nc10), axis=0)
                if normalSubjects[x] == 6:
                    XTrainF2 = np.concatenate((XTrainF2, n11F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, n11C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, n11F4), axis=0)
                    YTrain = np.concatenate((YTrain, nc11), axis=0)
                if normalSubjects[x] == 7:
                    XTrainF2 = np.concatenate((XTrainF2, n16F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, n16C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, n16F4), axis=0)
                    YTrain = np.concatenate((YTrain, nc16), axis=0)
                if normalSubjects[x] == 8:
                    XTrainF2 = np.concatenate((XTrainF2, nfle1F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, nfle1C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, nfle1F4), axis=0)
                    YTrain = np.concatenate((YTrain, nflec1), axis=0)
                if normalSubjects[x] == 9:
                    XTrainF2 = np.concatenate((XTrainF2, nfle2F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, nfle2C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, nfle2F4), axis=0)
                    YTrain = np.concatenate((YTrain, nflec2), axis=0)
                if normalSubjects[x] == 10:
                    XTrainF2 = np.concatenate((XTrainF2, nfle4F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, nfle4C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, nfle4F4), axis=0)
                    YTrain = np.concatenate((YTrain, nflec4), axis=0)
                if normalSubjects[x] == 11:
                    XTrainF2 = np.concatenate((XTrainF2, nfle7F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, nfle7C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, nfle7F4), axis=0)
                    YTrain = np.concatenate((YTrain, nflec7), axis=0)
                if normalSubjects[x] == 12:
                    XTrainF2 = np.concatenate((XTrainF2, nfle12F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, nfle12C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, nfle12F4), axis=0)
                    YTrain = np.concatenate((YTrain, nflec12), axis=0)
                if normalSubjects[x] == 13:
                    XTrainF2 = np.concatenate((XTrainF2, nfle13F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, nfle13C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, nfle13F4), axis=0)
                    YTrain = np.concatenate((YTrain, nflec13), axis=0)
                if normalSubjects[x] == 14:
                    XTrainF2 = np.concatenate((XTrainF2, nfle14F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, nfle14C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, nfle14F4), axis=0)
                    YTrain = np.concatenate((YTrain, nflec14), axis=0)
                if normalSubjects[x] == 15:
                    XTrainF2 = np.concatenate((XTrainF2, nfle15F2), axis=0)
                    XTrainC4 = np.concatenate((XTrainC4, nfle15C4), axis=0)
                    XTrainF4 = np.concatenate((XTrainF4, nfle15F4), axis=0)
                    YTrain = np.concatenate((YTrain, nflec15), axis=0)
        XTrainF2 = XTrainF2.reshape(round(len(XTrainF2) / 100), 100)
        XTrainC4 = XTrainC4.reshape(round(len(XTrainC4) / 100), 100)
        XTrainF4 = XTrainF4.reshape(round(len(XTrainF4) / 100), 100)
        XTestF2 = XTestF2.reshape(round(len(XTestF2) / 100), 100)
        XTestC4 = XTestC4.reshape(round(len(XTestC4) / 100), 100)
        XTestF4 = XTestF4.reshape(round(len(XTestF4) / 100), 100)
        """
        Clear the memory
        """
        del n1F2, n1C4, n1F4, nc1
        del n2F2, n2C4, n2F4, nc2
        del n3F2, n3C4, n3F4, nc3
        del n4F2, n4C4, n4F4, nc4
        del n5F2, n5C4, n5F4, nc5
        del n10F2, n10C4, n10F4, nc10
        del n11F2, n11C4, n11F4, nc11
        del n16F2, n16C4, n16F4, nc16
        del nfle1F2, nfle1C4, nfle1F4, nflec1
        del nfle2F2, nfle2C4, nfle2F4, nflec2
        del nfle4F2, nfle4C4, nfle4F4, nflec4
        del nfle7F2, nfle7C4, nfle7F4, nflec7
        del nfle12F2, nfle12C4, nfle12F4, nflec12
        del nfle13F2, nfle13C4, nfle13F4, nflec13
        del nfle14F2, nfle14C4, nfle14F4, nflec14
        del nfle15F2, nfle15C4, nfle15F4, nflec15
        gc.collect()
        tf.keras.backend.clear_session()
        """
        Create the labels data
        """
        for i in range(0, len(YTrain), 1):  # just A phase
            if YTrain[i] > 0:
                YTrain[i] = 1
            else:
                YTrain[i] = 0
        for i in range(0, len(YTest), 1):  # just A phase
            if YTest[i] > 0:
                YTest[i] = 1
            else:
                YTest[i] = 0
        """
        Estimate the class weights
        """
        from sklearn.utils import class_weight

        class_weights = class_weight.compute_class_weight('balanced', np.unique(YTrain), YTrain)
        """
        Produce the time steps data by expanding the dimentions
        """
        if NumberOfTimeSteps == 10:
            timeSteps = 10
            for ii in range(timeSteps - 1):  # delete first labels because of the overlapping time step
                YTrain = np.delete(YTrain, 0, 0)
                YTest = np.delete(YTest, 0, 0)
            encoder = LabelEncoder()
            encoder.fit(YTrain)
            encoded_YTrain = encoder.transform(YTrain)
            dummy_YTrain = np_utils.to_categorical(encoded_YTrain, 2)
            encoder = LabelEncoder()
            encoder.fit(YTest)
            encoded_YTest = encoder.transform(YTest)
            dummy_YTest = np_utils.to_categorical(encoded_YTest, 2)
            YTrain = dummy_YTrain
            YTest = dummy_YTest
            XTrainF2 = np.repeat(XTrainF2, timeSteps, axis=0)
            XTestF2 = np.repeat(XTestF2, timeSteps, axis=0)
            XTrainF22 = np.zeros((int(len(XTrainF2) - timeSteps * 9), features))
            indi = 1
            for k in range(len(XTrainF2) - timeSteps * 9):
                if indi == 1:
                    XTrainF22[k, :] = XTrainF2[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 8 - 1, :]
                    indi = 10
                else:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 9 - 1, :]
                    indi = 1
            XTrainF2 = XTrainF22.reshape(round(XTrainF22.shape[0] / timeSteps), timeSteps, features)
            XTestF22 = np.zeros((int(len(XTestF2) - timeSteps * 9), features))
            indi = 1
            for k in range(len(XTestF2) - timeSteps * 9):
                if indi == 1:
                    XTestF22[k, :] = XTestF2[k, :]
                    indi = 2
                elif indi == 2:
                    XTestF22[k, :] = XTestF2[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 8 - 1, :]
                    indi = 10
                else:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 9 - 1, :]
                    indi = 1
            XTestF2 = XTestF22.reshape(round(XTestF22.shape[0] / timeSteps), timeSteps, features)
            XTrainC4 = np.repeat(XTrainC4, timeSteps, axis=0)
            XTestC4 = np.repeat(XTestC4, timeSteps, axis=0)
            XTrainC42 = np.zeros((int(len(XTrainC4) - timeSteps * 9), features))
            indi = 1
            for k in range(len(XTrainC4) - timeSteps * 9):
                if indi == 1:
                    XTrainC42[k, :] = XTrainC4[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 8 - 1, :]
                    indi = 10
                else:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 9 - 1, :]
                    indi = 1
            XTrainC4 = XTrainC42.reshape(round(XTrainC42.shape[0] / timeSteps), timeSteps, features)
            XTestC42 = np.zeros((int(len(XTestC4) - timeSteps * 9), features))
            indi = 1
            for k in range(len(XTestC4) - timeSteps * 9):
                if indi == 1:
                    XTestC42[k, :] = XTestC4[k, :]
                    indi = 2
                elif indi == 2:
                    XTestC42[k, :] = XTestC4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 8 - 1, :]
                    indi = 10
                else:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 9 - 1, :]
                    indi = 1
            XTestC4 = XTestC42.reshape(round(XTestC42.shape[0] / timeSteps), timeSteps, features)
            XTrainF4 = np.repeat(XTrainF4, timeSteps, axis=0)
            XTestF4 = np.repeat(XTestF4, timeSteps, axis=0)
            XTrainF42 = np.zeros((int(len(XTrainF4) - timeSteps * 9), features))
            indi = 1
            for k in range(len(XTrainF4) - timeSteps * 9):
                if indi == 1:
                    XTrainF42[k, :] = XTrainF4[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 8 - 1, :]
                    indi = 10
                else:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 9 - 1, :]
                    indi = 1
            XTrainF4 = XTrainF42.reshape(round(XTrainF42.shape[0] / timeSteps), timeSteps, features)
            XTestF42 = np.zeros((int(len(XTestF4) - timeSteps * 9), features))
            indi = 1
            for k in range(len(XTestF4) - timeSteps * 9):
                if indi == 1:
                    XTestF42[k, :] = XTestF4[k, :]
                    indi = 2
                elif indi == 2:
                    XTestF42[k, :] = XTestF4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 8 - 1, :]
                    indi = 10
                else:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 9 - 1, :]
                    indi = 1
            XTestF4 = XTestF42.reshape(round(XTestF42.shape[0] / timeSteps), timeSteps, features)
        elif NumberOfTimeSteps == 15:
            timeSteps = 15
            for ii in range(timeSteps - 1):  # delete first labels because of the overlapping time step
                YTrain = np.delete(YTrain, 0, 0)
                YTest = np.delete(YTest, 0, 0)
            encoder = LabelEncoder()
            encoder.fit(YTrain)
            encoded_YTrain = encoder.transform(YTrain)
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_YTrain = np_utils.to_categorical(encoded_YTrain, 2)
            encoder = LabelEncoder()
            encoder.fit(YTest)
            encoded_YTest = encoder.transform(YTest)
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_YTest = np_utils.to_categorical(encoded_YTest, 2)
            YTrain = dummy_YTrain
            YTest = dummy_YTest
            XTrainF2 = np.repeat(XTrainF2, timeSteps, axis=0)
            XTestF2 = np.repeat(XTestF2, timeSteps, axis=0)
            XTrainF22 = np.zeros((int(len(XTrainF2) - timeSteps * 14), features))
            indi = 1
            for k in range(len(XTrainF2) - timeSteps * 14):
                if indi == 1:
                    XTrainF22[k, :] = XTrainF2[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 13 - 1, :]
                    indi = 15
                else:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 14 - 1, :]
                    indi = 1
            XTrainF2 = XTrainF22.reshape(round(XTrainF22.shape[0] / timeSteps), timeSteps, features)
            XTestF22 = np.zeros((int(len(XTestF2) - timeSteps * 14), features))
            indi = 1
            for k in range(len(XTestF2) - timeSteps * 14):
                if indi == 1:
                    XTestF22[k, :] = XTestF2[k, :]
                    indi = 2
                elif indi == 2:
                    XTestF22[k, :] = XTestF2[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 13 - 1, :]
                    indi = 15
                else:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 14 - 1, :]
                    indi = 1
            XTestF2 = XTestF22.reshape(round(XTestF22.shape[0] / timeSteps), timeSteps, features)
            XTrainC4 = np.repeat(XTrainC4, timeSteps, axis=0)
            XTestC4 = np.repeat(XTestC4, timeSteps, axis=0)
            XTrainC42 = np.zeros((int(len(XTrainC4) - timeSteps * 14), features))
            indi = 1
            for k in range(len(XTrainC4) - timeSteps * 14):
                if indi == 1:
                    XTrainC42[k, :] = XTrainC4[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 13 - 1, :]
                    indi = 15
                else:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 14 - 1, :]
                    indi = 1
            XTrainC4 = XTrainC42.reshape(round(XTrainC42.shape[0] / timeSteps), timeSteps, features)
            XTestC42 = np.zeros((int(len(XTestC4) - timeSteps * 14), features))
            indi = 1
            for k in range(len(XTestC4) - timeSteps * 14):
                if indi == 1:
                    XTestC42[k, :] = XTestC4[k, :]
                    indi = 2
                elif indi == 2:
                    XTestC42[k, :] = XTestC4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 13 - 1, :]
                    indi = 15
                else:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 14 - 1, :]
                    indi = 1
            XTestC4 = XTestC42.reshape(round(XTestC42.shape[0] / timeSteps), timeSteps, features)
            XTrainF4 = np.repeat(XTrainF4, timeSteps, axis=0)
            XTestF4 = np.repeat(XTestF4, timeSteps, axis=0)
            XTrainF42 = np.zeros((int(len(XTrainF4) - timeSteps * 14), features))
            indi = 1
            for k in range(len(XTrainF4) - timeSteps * 14):
                if indi == 1:
                    XTrainF42[k, :] = XTrainF4[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 13 - 1, :]
                    indi = 15
                else:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 14 - 1, :]
                    indi = 1
            XTrainF4 = XTrainF42.reshape(round(XTrainF42.shape[0] / timeSteps), timeSteps, features)
            XTestF42 = np.zeros((int(len(XTestF4) - timeSteps * 14), features))
            indi = 1
            for k in range(len(XTestF4) - timeSteps * 14):
                if indi == 1:
                    XTestF42[k, :] = XTestF4[k, :]
                    indi = 2
                elif indi == 2:
                    XTestF42[k, :] = XTestF4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 13 - 1, :]
                    indi = 15
                else:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 14 - 1, :]
                    indi = 1
            XTestF4 = XTestF42.reshape(round(XTestF42.shape[0] / timeSteps), timeSteps, features)
        elif NumberOfTimeSteps == 20:
            timeSteps = 20
            for ii in range(timeSteps - 1):  # delete first labels because of the overlapping time step
                YTrain = np.delete(YTrain, 0, 0)
                YTest = np.delete(YTest, 0, 0)
            encoder = LabelEncoder()
            encoder.fit(YTrain)
            encoded_YTrain = encoder.transform(YTrain)
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_YTrain = np_utils.to_categorical(encoded_YTrain, 2)
            encoder = LabelEncoder()
            encoder.fit(YTest)
            encoded_YTest = encoder.transform(YTest)
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_YTest = np_utils.to_categorical(encoded_YTest, 2)
            YTrain = dummy_YTrain
            YTest = dummy_YTest
            XTrainF2 = np.repeat(XTrainF2, timeSteps, axis=0)
            XTestF2 = np.repeat(XTestF2, timeSteps, axis=0)
            XTrainF22 = np.zeros((int(len(XTrainF2) - timeSteps * 19), features))
            indi = 1
            for k in range(len(XTrainF2) - timeSteps * 19):
                if indi == 1:
                    XTrainF22[k, :] = XTrainF2[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 18 - 1, :]
                    indi = 20
                else:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 19 - 1, :]
                    indi = 1
            XTrainF2 = XTrainF22.reshape(round(XTrainF22.shape[0] / timeSteps), timeSteps, features)
            XTestF22 = np.zeros((int(len(XTestF2) - timeSteps * 19), features))
            indi = 1
            for k in range(len(XTestF2) - timeSteps * 19):
                if indi == 1:
                    XTestF22[k, :] = XTestF2[k, :]
                    indi = 2
                elif indi == 2:
                    XTestF22[k, :] = XTestF2[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 18 - 1, :]
                    indi = 20
                else:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 19 - 1, :]
                    indi = 1
            XTestF2 = XTestF22.reshape(round(XTestF22.shape[0] / timeSteps), timeSteps, features)
            XTrainC4 = np.repeat(XTrainC4, timeSteps, axis=0)
            XTestC4 = np.repeat(XTestC4, timeSteps, axis=0)
            XTrainC42 = np.zeros((int(len(XTrainC4) - timeSteps * 19), features))
            indi = 1
            for k in range(len(XTrainC4) - timeSteps * 19):
                if indi == 1:
                    XTrainC42[k, :] = XTrainC4[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 18 - 1, :]
                    indi = 20
                else:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 19 - 1, :]
                    indi = 1
            XTrainC4 = XTrainC42.reshape(round(XTrainC42.shape[0] / timeSteps), timeSteps, features)
            XTestC42 = np.zeros((int(len(XTestC4) - timeSteps * 19), features))
            indi = 1
            for k in range(len(XTestC4) - timeSteps * 19):
                if indi == 1:
                    XTestC42[k, :] = XTestC4[k, :]
                    indi = 2
                elif indi == 2:
                    XTestC42[k, :] = XTestC4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 18 - 1, :]
                    indi = 20
                else:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 19 - 1, :]
                    indi = 1
            XTestC4 = XTestC42.reshape(round(XTestC42.shape[0] / timeSteps), timeSteps, features)
            XTrainF4 = np.repeat(XTrainF4, timeSteps, axis=0)
            XTestF4 = np.repeat(XTestF4, timeSteps, axis=0)
            XTrainF42 = np.zeros((int(len(XTrainF4) - timeSteps * 19), features))
            indi = 1
            for k in range(len(XTrainF4) - timeSteps * 19):
                if indi == 1:
                    XTrainF42[k, :] = XTrainF4[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 18 - 1, :]
                    indi = 20
                else:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 19 - 1, :]
                    indi = 1
            XTrainF4 = XTrainF42.reshape(round(XTrainF42.shape[0] / timeSteps), timeSteps, features)
            XTestF42 = np.zeros((int(len(XTestF4) - timeSteps * 19), features))
            indi = 1
            for k in range(len(XTestF4) - timeSteps * 19):
                if indi == 1:
                    XTestF42[k, :] = XTestF4[k, :]
                    indi = 2
                elif indi == 2:
                    XTestF42[k, :] = XTestF4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 18 - 1, :]
                    indi = 20
                else:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 19 - 1, :]
                    indi = 1
            XTestF4 = XTestF42.reshape(round(XTestF42.shape[0] / timeSteps), timeSteps, features)
        else:
            timeSteps = 25
            for ii in range(timeSteps - 1):  # delete first labels because of the overlapping time step
                YTrain = np.delete(YTrain, 0, 0)
                YTest = np.delete(YTest, 0, 0)
            encoder = LabelEncoder()
            encoder.fit(YTrain)
            encoded_YTrain = encoder.transform(YTrain)
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_YTrain = np_utils.to_categorical(encoded_YTrain, 2)
            encoder = LabelEncoder()
            encoder.fit(YTest)
            encoded_YTest = encoder.transform(YTest)
            # convert integers to dummy variables (i.e. one hot encoded)
            dummy_YTest = np_utils.to_categorical(encoded_YTest, 2)
            YTrain = dummy_YTrain
            YTest = dummy_YTest
            XTrainF2 = np.repeat(XTrainF2, timeSteps, axis=0)
            XTestF2 = np.repeat(XTestF2, timeSteps, axis=0)
            XTrainF22 = np.zeros((int(len(XTrainF2) - timeSteps * 24), features))
            indi = 1
            for k in range(len(XTrainF2) - timeSteps * 24):
                if indi == 1:
                    XTrainF22[k, :] = XTrainF2[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 23 - 1, :]
                    indi = 25
                else:
                    XTrainF22[k, :] = XTrainF2[k + timeSteps * 24 - 1, :]
                    indi = 1
            XTrainF2 = XTrainF22.reshape(round(XTrainF22.shape[0] / timeSteps), timeSteps, features)
            XTestF22 = np.zeros((int(len(XTestF2) - timeSteps * 24), features))
            indi = 1
            for k in range(len(XTestF2) - timeSteps * 24):
                if indi == 1:
                    XTestF22[k, :] = XTestF2[k, :]
                    indi = 2
                elif indi == 2:
                    XTestF22[k, :] = XTestF2[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 23 - 1, :]
                    indi = 25
                else:
                    XTestF22[k, :] = XTestF2[k + timeSteps * 24 - 1, :]
                    indi = 1
            XTestF2 = XTestF22.reshape(round(XTestF22.shape[0] / timeSteps), timeSteps, features)
            XTrainC4 = np.repeat(XTrainC4, timeSteps, axis=0)
            XTestC4 = np.repeat(XTestC4, timeSteps, axis=0)
            XTrainC42 = np.zeros((int(len(XTrainC4) - timeSteps * 24), features))
            indi = 1
            for k in range(len(XTrainC4) - timeSteps * 24):
                if indi == 1:
                    XTrainC42[k, :] = XTrainC4[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 23 - 1, :]
                    indi = 25
                else:
                    XTrainC42[k, :] = XTrainC4[k + timeSteps * 24 - 1, :]
                    indi = 1
            XTrainC4 = XTrainC42.reshape(round(XTrainC42.shape[0] / timeSteps), timeSteps, features)
            XTestC42 = np.zeros((int(len(XTestC4) - timeSteps * 24), features))
            indi = 1
            for k in range(len(XTestC4) - timeSteps * 24):
                if indi == 1:
                    XTestC42[k, :] = XTestC4[k, :]
                    indi = 2
                elif indi == 2:
                    XTestC42[k, :] = XTestC4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 23 - 1, :]
                    indi = 25
                else:
                    XTestC42[k, :] = XTestC4[k + timeSteps * 24 - 1, :]
                    indi = 1
            XTestC4 = XTestC42.reshape(round(XTestC42.shape[0] / timeSteps), timeSteps, features)
            XTrainF4 = np.repeat(XTrainF4, timeSteps, axis=0)
            XTestF4 = np.repeat(XTestF4, timeSteps, axis=0)
            XTrainF42 = np.zeros((int(len(XTrainF4) - timeSteps * 24), features))
            indi = 1
            for k in range(len(XTrainF4) - timeSteps * 24):
                if indi == 1:
                    XTrainF42[k, :] = XTrainF4[k, :]
                    indi = 2
                elif indi == 2:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 23 - 1, :]
                    indi = 25
                else:
                    XTrainF42[k, :] = XTrainF4[k + timeSteps * 24 - 1, :]
                    indi = 1
            XTrainF4 = XTrainF42.reshape(round(XTrainF42.shape[0] / timeSteps), timeSteps, features)
            XTestF42 = np.zeros((int(len(XTestF4) - timeSteps * 24), features))
            indi = 1
            for k in range(len(XTestF4) - timeSteps * 24):
                if indi == 1:
                    XTestF42[k, :] = XTestF4[k, :]
                    indi = 2
                elif indi == 2:
                    XTestF42[k, :] = XTestF4[k + timeSteps - 1, :]
                    indi = 3
                elif indi == 3:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 2 - 1, :]
                    indi = 4
                elif indi == 4:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 3 - 1, :]
                    indi = 5
                elif indi == 5:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 4 - 1, :]
                    indi = 6
                elif indi == 6:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 5 - 1, :]
                    indi = 7
                elif indi == 7:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 6 - 1, :]
                    indi = 8
                elif indi == 8:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 7 - 1, :]
                    indi = 9
                elif indi == 9:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 8 - 1, :]
                    indi = 10
                elif indi == 10:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 9 - 1, :]
                    indi = 11
                elif indi == 11:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 10 - 1, :]
                    indi = 12
                elif indi == 12:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 11 - 1, :]
                    indi = 13
                elif indi == 13:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 12 - 1, :]
                    indi = 14
                elif indi == 14:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 13 - 1, :]
                    indi = 15
                elif indi == 15:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 14 - 1, :]
                    indi = 16
                elif indi == 16:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 15 - 1, :]
                    indi = 17
                elif indi == 17:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 16 - 1, :]
                    indi = 18
                elif indi == 18:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 17 - 1, :]
                    indi = 19
                elif indi == 19:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 18 - 1, :]
                    indi = 20
                elif indi == 20:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 19 - 1, :]
                    indi = 21
                elif indi == 21:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 20 - 1, :]
                    indi = 22
                elif indi == 22:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 21 - 1, :]
                    indi = 23
                elif indi == 23:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 22 - 1, :]
                    indi = 24
                elif indi == 24:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 23 - 1, :]
                    indi = 25
                else:
                    XTestF42[k, :] = XTestF4[k + timeSteps * 24 - 1, :]
                    indi = 1
            XTestF4 = XTestF42.reshape(round(XTestF42.shape[0] / timeSteps), timeSteps, features)
        """
        Configure the classifier's structure
        """
        inp1 = Input(shape=(XTrainF2.shape[1], XTrainF2.shape[2]))
        inp2 = Input(shape=(XTrainC4.shape[1], XTrainC4.shape[2]))
        inp3 = Input(shape=(XTrainF4.shape[1], XTrainF4.shape[2]))
        if NumberOfLSTMLayers == 1:
            if TypeLSTMLayer == 0:
                x = LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout)(inp1)
                y = LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout)(inp2)
                z = LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout)(inp3)
            else:
                x = Bidirectional(LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout))(inp1)
                y = Bidirectional(LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout))(inp2)
                z = Bidirectional(LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout))(inp3)
        else:
            if TypeLSTMLayer == 0:
                x = LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout, return_sequences=True)(inp1)
                y = LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout, return_sequences=True)(inp2)
                z = LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout, return_sequences=True)(inp3)
                x = LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout)(x)
                y = LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout)(y)
                z = LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout)(z)
            else:
                x = Bidirectional(LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout, return_sequences=True))(inp1)
                y = Bidirectional(LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout, return_sequences=True))(inp2)
                z = Bidirectional(LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout, return_sequences=True))(inp3)
                x = Bidirectional(LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout))(x)
                y = Bidirectional(LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout))(y)
                z = Bidirectional(LSTM(ShapeOfLSTM, dropout=PercentageDropout, recurrent_dropout=PercentageDropout))(z)
        if ShapeOfDenseLayer > 0:
            if ActivationFunction == 0:
                x = Dense(ShapeOfDenseLayer, activation='tanh')(x)
                x = Dropout(PercentageDropout)(x)
                y = Dense(ShapeOfDenseLayer, activation='tanh')(y)
                y = Dropout(PercentageDropout)(y)
                z = Dense(ShapeOfDenseLayer, activation='tanh')(z)
                z = Dropout(PercentageDropout)(z)
            elif ActivationFunction == 1:
                x = Dense(ShapeOfDenseLayer, activation='sigmoid')(x)
                x = Dropout(PercentageDropout)(x)
                y = Dense(ShapeOfDenseLayer, activation='sigmoid')(y)
                y = Dropout(PercentageDropout)(y)
                z = Dense(ShapeOfDenseLayer, activation='sigmoid')(z)
                z = Dropout(PercentageDropout)(z)
            elif ActivationFunction == 2:
                x = Dense(ShapeOfDenseLayer, activation='relu')(x)
                x = Dropout(PercentageDropout)(x)
                y = Dense(ShapeOfDenseLayer, activation='relu')(y)
                y = Dropout(PercentageDropout)(y)
                z = Dense(ShapeOfDenseLayer, activation='relu')(z)
                z = Dropout(PercentageDropout)(z)
            elif ActivationFunction == 3:
                x = Dense(ShapeOfDenseLayer, activation='selu')(x)
                x = Dropout(PercentageDropout)(x)
                y = Dense(ShapeOfDenseLayer, activation='selu')(y)
                y = Dropout(PercentageDropout)(y)
                z = Dense(ShapeOfDenseLayer, activation='selu')(z)
                z = Dropout(PercentageDropout)(z)
        if NumberOfCannels == 1:
            out = Dense(2, activation='softmax')(x)
            model = Model(inputs=inp1, outputs=out)
        elif NumberOfCannels == 2:
            out = Dense(2, activation='softmax')(y)
            model = Model(inputs=inp2, outputs=out)
        elif NumberOfCannels == 3:
            out = Dense(2, activation='softmax')(z)
            model = Model(inputs=inp3, outputs=out)
        elif NumberOfCannels == 4:
            w = concatenate([x, y])
            if ShapeOfDenseLayer > 0:
                if ActivationFunction == 0:
                    w = Dense(ShapeOfDenseLayer, activation='tanh')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 1:
                    w = Dense(ShapeOfDenseLayer, activation='sigmoid')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 2:
                    w = Dense(ShapeOfDenseLayer, activation='relu')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 3:
                    w = Dense(ShapeOfDenseLayer, activation='selu')(w)
                    w = Dropout(PercentageDropout)(w)
            out = Dense(2, activation='softmax')(w)
            model = Model(inputs=[inp1, inp2], outputs=out)
        elif NumberOfCannels == 5:
            w = concatenate([x, z])
            if ShapeOfDenseLayer > 0:
                if ActivationFunction == 0:
                    w = Dense(ShapeOfDenseLayer, activation='tanh')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 1:
                    w = Dense(ShapeOfDenseLayer, activation='sigmoid')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 2:
                    w = Dense(ShapeOfDenseLayer, activation='relu')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 3:
                    w = Dense(ShapeOfDenseLayer, activation='selu')(w)
                    w = Dropout(PercentageDropout)(w)
            out = Dense(2, activation='softmax')(w)
            model = Model(inputs=[inp1, inp3], outputs=out)
        elif NumberOfCannels == 6:
            w = concatenate([y, z])
            if ShapeOfDenseLayer > 0:
                if ActivationFunction == 0:
                    w = Dense(ShapeOfDenseLayer, activation='tanh')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 1:
                    w = Dense(ShapeOfDenseLayer, activation='sigmoid')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 2:
                    w = Dense(ShapeOfDenseLayer, activation='relu')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 3:
                    w = Dense(ShapeOfDenseLayer, activation='selu')(w)
                    w = Dropout(PercentageDropout)(w)
            out = Dense(2, activation='softmax')(w)
            model = Model(inputs=[inp2, inp3], outputs=out)
        else:
            w = concatenate([x, y, z])
            if ShapeOfDenseLayer > 0:
                if ActivationFunction == 0:
                    w = Dense(ShapeOfDenseLayer, activation='tanh')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 1:
                    w = Dense(ShapeOfDenseLayer, activation='sigmoid')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 2:
                    w = Dense(ShapeOfDenseLayer, activation='relu')(w)
                    w = Dropout(PercentageDropout)(w)
                elif ActivationFunction == 3:
                    w = Dense(ShapeOfDenseLayer, activation='selu')(w)
                    w = Dropout(PercentageDropout)(w)
            out = Dense(2, activation='softmax')(w)
            model = Model(inputs=[inp1, inp2, inp3], outputs=out)
        """
        Class to perform the early stopping
        """


        class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
            a = 1


        class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
            def __init__(self, patience=patience):
                super(EarlyStoppingAtMinLoss, self).__init__()
                self.patience = patience
                self.best_weights = None

            def on_train_begin(self, logs=None):
                self.wait = 0
                self.stopped_epoch = 0
                self.best = 0.2
                self._data = []
                self.curentAUC = 0.2

            def on_epoch_end(self, epoch, logs=None):
                if NumberOfCannels < 4:
                    X_val1 = self.validation_data[0]
                    y_val = self.validation_data[1]
                    y_predict = np.asarray(model.predict(X_val1))
                elif NumberOfCannels < 7:
                    X_val1 = self.validation_data[0]
                    X_val2 = self.validation_data[1]
                    y_val = self.validation_data[2]
                    y_predict = np.asarray(model.predict([X_val1, X_val2]))
                else:
                    X_val1 = self.validation_data[0]
                    X_val2 = self.validation_data[1]
                    X_val3 = self.validation_data[2]
                    y_val = self.validation_data[3]
                    y_predict = np.asarray(model.predict([X_val1, X_val2, X_val3]))
                fpr_keras, tpr_keras, thresholds_keras = roc_curve(np.argmax(y_val, axis=1), y_predict[:, 1])
                auc_keras = auc(fpr_keras, tpr_keras)
                self.curentAUC = auc_keras
                if np.greater(self.curentAUC, self.best + 0.005):  # np.less
                    self.best = self.curentAUC
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        self.model.set_weights(self.best_weights)

            def on_train_end(self, logs=None):
                if self.stopped_epoch > 0:
                    a = 1


        """
        Compile the classifier
        """
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
        """
        Train the classifier
        """
        if NumberOfCannels == 1:
            x_train, x_valid, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
            model.fit(x_train, y_train,
                      batch_size=BatchSize,
                      epochs=100,  # 10000
                      validation_data=(x_valid, y_valid),
                      verbose=0,
                      class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
            #            print("Testing epoch", ee)
            proba = model.predict(XTestF2)
        elif NumberOfCannels == 2:
            x_train, x_valid, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
            model.fit(x_train, y_train,
                      batch_size=BatchSize,
                      epochs=100,  # 10000
                      validation_data=(x_valid, y_valid),
                      verbose=0,
                      class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
            #            print("Testing epoch", ee)
            proba = model.predict(XTestC4)
        elif NumberOfCannels == 3:
            x_train, x_valid, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
            model.fit(x_train, y_train,
                      batch_size=BatchSize,
                      epochs=100,  # 10000
                      validation_data=(x_valid, y_valid),
                      verbose=0,
                      class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
            #            print("Testing epoch", ee)
            proba = model.predict(XTestF4)
        elif NumberOfCannels == 4:
            x_trainF2, x_validF2, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
            x_trainC4, x_validC4, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
            model.fit([x_trainF2, x_trainC4], y_train,
                      batch_size=BatchSize,
                      epochs=100,  # 10000
                      validation_data=([x_validF2, x_validC4], y_valid),
                      verbose=0,
                      class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
            #            print("Testing epoch", ee)
            proba = model.predict([XTestF2, XTestC4])
        elif NumberOfCannels == 5:
            x_trainF2, x_validF2, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
            x_trainF4, x_validF4, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
            model.fit([x_trainF2, x_trainF4], y_train,
                      batch_size=BatchSize,
                      epochs=100,  # 10000
                      validation_data=([x_validF2, x_validF4], y_valid),
                      verbose=0,
                      class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
            #            print("Testing epoch", ee)
            proba = model.predict([XTestF2, XTestF4])
        elif NumberOfCannels == 6:
            x_trainC4, x_validC4, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
            x_trainF4, x_validF4, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
            model.fit([x_trainC4, x_trainF4], y_train,
                      batch_size=BatchSize,
                      epochs=100,  # 10000
                      validation_data=([x_validC4, x_validF4], y_valid),
                      verbose=0,
                      class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
            #            print("Testing epoch", ee)
            proba = model.predict([XTestC4, XTestF4])
        else:
            x_trainF2, x_validF2, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
            x_trainC4, x_validC4, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
            x_trainF4, x_validF4, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
            model.fit([x_trainF2, x_trainC4, x_trainF4], y_train,
                      batch_size=BatchSize,
                      epochs=100,  # 10000
                      validation_data=([x_validF2, x_validC4, x_validF4], y_valid),
                      verbose=0,
                      class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
            #            print("Testing epoch", ee)
            proba = model.predict([XTestF2, XTestC4, XTestF4])
        """
        Check the performance
        """
        YTestOneLine = np.zeros(len(YTest))
        for x in range(len(YTest)):
            if YTest[x, 0] == 1:
                YTestOneLine[x] = 0
            else:
                YTestOneLine[x] = 1
        predictiony_predMean = np.mean(proba[:, 0])
        predictiony_pred = np.zeros(len(YTestOneLine))

        YTrainOneLine = np.zeros(len(YTrain))
        for x in range(len(YTrain)):
            if YTrain[x, 0] == 1:
                YTrainOneLine[x] = 0
            else:
                YTrainOneLine[x] = 1
        probaTrain = model.predict([XTrainF2, XTrainC4, XTrainF4])
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(YTrainOneLine, probaTrain[:, 1])  # produce the receiving operating curve
        auc_keras = auc(fpr_keras, tpr_keras)  # estimate the AUC
        import pandas as pd

        i = np.arange(len(tpr_keras))
        roc = pd.DataFrame({'tf': pd.Series(tpr_keras - (1 - fpr_keras), index=i), 'threshold': pd.Series(thresholds_keras, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        TrhesholdForClassification = np.float(roc_t['threshold'])

        for x in range(len(YTestOneLine)):
            if proba[x, 1] > TrhesholdForClassification:
                predictiony_pred[x] = 1
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(YTestOneLine, proba[:, 1])
        auc_keras = auc(fpr_keras, tpr_keras)
        print('AUC : ', auc_keras)
        """
        Post processing
        """
        capPredictedPredicted = predictiony_pred
        for k in range(len(capPredictedPredicted) - 1):
            if k > 0:
                if capPredictedPredicted[k - 1] == 0 and capPredictedPredicted[k] == 1 and capPredictedPredicted[k + 1] == 0:
                    capPredictedPredicted[k] = 0

        for k in range(len(capPredictedPredicted) - 1):
            if k > 0:
                if capPredictedPredicted[k - 1] == 1 and capPredictedPredicted[k] == 0 and capPredictedPredicted[k + 1] == 1:
                    capPredictedPredicted[k] = 1
        """
        Check and save the LOO cycle results
        """
        tn, fp, fn, tp = confusion_matrix(YTestOneLine, capPredictedPredicted).ravel()
        print(classification_report(YTestOneLine, capPredictedPredicted))
        accuracy0 = (tp + tn) / (tp + tn + fp + fn)
        print('Accuracy : ', accuracy0)
        sensitivity0 = tp / (tp + fn)
        print('Sensitivity : ', sensitivity0)
        specificity0 = tn / (fp + tn)
        print('Specificity : ', specificity0)
        AccAtEnd[ff] = accuracy0
        SenAtEnd[ff] = sensitivity0
        SpeAtEnd[ff] = specificity0
        AUCAtEnd[ff] = auc_keras
        TPEnd[ff] = tp
        TNEnd[ff] = tn
        FPEnd[ff] = fp
        FNEnd[ff] = fn
        """
        Clear the memory
        """
        del XTrainF2, XTrainC4, XTrainF4, YTrain, XTestF2, XTestC4, XTestF4, YTest, model
    """
    Check and save the LOO results for the current subject
    """
    AccLOO[ee] = np.mean(AccAtEnd)
    SenLOO[ee] = np.mean(SenAtEnd)
    SpeLOO[ee] = np.mean(SpeAtEnd)
    AUCLOO[ee] = np.mean(AUCAtEnd)
    TPLOO[ee] = np.mean(TPEnd)
    TNLOO[ee] = np.mean(TNEnd)
    FPLOO[ee] = np.mean(FPEnd)
    FNLOO[ee] = np.mean(FNEnd)
    f = open("AccClass " + str(ee) + ".txt", 'ab')
    pickle.dump(AccLOO, f)
    f.close()
    f = open("SenClass " + str(ee) + ".txt", 'ab')
    pickle.dump(SenLOO, f)
    f.close()
    f = open("SpeClass " + str(ee) + ".txt", 'ab')
    pickle.dump(SpeLOO, f)
    f.close()
    f = open("AUCClass " + str(ee) + ".txt", 'ab')
    pickle.dump(AUCLOO, f)
    f.close()
    f = open("TP " + str(ee) + ".txt", 'ab')
    pickle.dump(TPLOO, f)
    f.close()
    f = open("TN " + str(ee) + ".txt", 'ab')
    pickle.dump(TNLOO, f)
    f.close()
    f = open("FP " + str(ee) + ".txt", 'ab')
    pickle.dump(FPLOO, f)
    f.close()
    f = open("FN " + str(ee) + ".txt", 'ab')
    pickle.dump(FNLOO, f)
    f.close()
