import gc
import pickle
import random
import time

import numpy as np
import scipy.io as spio
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.layers import Input, concatenate, Bidirectional
from keras.layers import LSTM
from keras.models import Model
from keras.utils import np_utils
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#############################
# Load and standardize data #
#############################

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

#############################
# Control variablees        #
#############################

Begin = 0
BeginVal = 8
BeginTest = 8
numberSubjects = 16
features = 100
Epochs = 2
patienteceValue = 10
BatchSize = 1000


def initilization_of_population(size, n_feat):
    """
    Method for initializing the population
    """
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat)
        chromosome[:int(0.5 * n_feat)] = 0
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population, GenerationsFile):
    """
    Method for defining the fitness function to optimize
    """

    scores = []
    indicationsGen = 0
    for chromosome in population:
        tf.keras.backend.clear_session()

        # Number of channels to be fused: 0-2
        # Number of time steps to be considered by the LSTM: 3-4
        # Number of LSTM layers for each channel: 5
        # Type of LSTM: 6
        # Shape of the LSTM layers: 7-8
        # Percentage of dropout for the recurrent and dense layers: 9-10
        # Shape of the dense layers: 11-12
        # Activation function for the dense layers: 13-14

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
            TypeLSTMLayer = 0  # LSTM
        else:
            TypeLSTMLayer = 1  # BLSTM

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

        AccAtEnd = np.zeros(Epochs)
        SenAtEnd = np.zeros(Epochs)
        SpeAtEnd = np.zeros(Epochs)
        AUCAtEnd = np.zeros(Epochs)
        TPEnd = np.zeros(Epochs)
        TNEnd = np.zeros(Epochs)
        FPEnd = np.zeros(Epochs)
        FNEnd = np.zeros(Epochs)

        #        # NumberOfCannels = 1 -> F2 000
        #        # NumberOfCannels = 2 -> C4 001
        #        # NumberOfCannels = 3 -> F4 010
        #        # NumberOfCannels = 4 -> F2 and C4 011
        #        # NumberOfCannels = 5 -> F2 and F4 100
        #        # NumberOfCannels = 6 -> C4 and F4 101
        #        # NumberOfCannels = 7 -> F2 and C4 and F4 110

        # Number of required bits: 15 b

        ########################################## The following lines are just for testing, comment when doing real simmulations:

        ## less computational demanding network
        #        NumberOfCannels=1
        #        NumberOfTimeSteps=5 # choose either 5, 15 or 25
        #        NumberOfLSTMLayers=1 # choose either 1 or 2
        #        NumberOfDenseLayersAfterLSTM=0 # choose either 0 or 1
        #        NumberOfDenseLayersAfterConcatenation=0 # choose either 0 or 1
        #        ShapeOfLSTMLayerF2=50 # choose either 100, 200, 300 or 400
        #        ShapeOfLSTMLayerC4=50 # choose either 100, 200, 300 or 400
        #        ShapeOfLSTMLayerF4=50 # choose either 100, 200, 300 or 400
        #        ShapeOfDenseLayerF2=50 # choose either 100, 200, 300 or 400
        #        ShapeOfDenseLayerC4=50 # choose either 100, 200, 300 or 400
        #        ShapeOfDenseLayerF4=50 # choose either 100, 200, 300 or 400
        #        ShapeOfDenseLayerAfterConcatenation=50 # choose either 100, 200, 300 or 400

        ## most computational demanding network
        #        NumberOfCannels=3
        #        NumberOfTimeSteps=25 # choose either 5, 15 or 25
        #        NumberOfLSTMLayers=2 # choose either 1 or 2
        #        NumberOfDenseLayersAfterLSTM=1 # choose either 0 or 1
        #        NumberOfDenseLayersAfterConcatenation=1 # choose either 0 or 1
        #        ShapeOfLSTMLayerF2=400 # choose either 100, 200, 300 or 400
        #        ShapeOfLSTMLayerC4=400 # choose either 100, 200, 300 or 400
        #        ShapeOfLSTMLayerF4=400 # choose either 100, 200, 300 or 400
        #        ShapeOfDenseLayerF2=400 # choose either 100, 200, 300 or 400
        #        ShapeOfDenseLayerC4=400 # choose either 100, 200, 300 or 400
        #        ShapeOfDenseLayerF4=400 # choose either 100, 200, 300 or 400
        #        ShapeOfDenseLayerAfterConcatenation=400 # choose either 100, 200, 300 or 400

        ####################################################################################################

        for ee in range(Epochs):
            tf.keras.backend.clear_session()
            if ee == 0:
                gc.collect()
                tf.keras.backend.clear_session()

                normalSubjects = np.random.permutation(numberSubjects)  # choose subjects order

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
                    # YTestHypno=nh1
                elif normalSubjects[BeginTest] == 1:
                    XTestF2 = n2F2
                    XTestC4 = n2C4
                    XTestF4 = n2F4
                    YTest = nc2
                    # YTestHypno=nh2
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
                    if x < BeginTest and x > Begin:  # train
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

                    if x <= numberSubjects - 1 and x >= BeginTest:  # test
                        if normalSubjects[x] == 0:
                            XTestF2 = np.concatenate((XTestF2, n1F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n1C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n1F4), axis=0)
                            YTest = np.concatenate((YTest, nc1), axis=0)
                        if normalSubjects[x] == 1:
                            XTestF2 = np.concatenate((XTestF2, n2F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n2C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n2F4), axis=0)
                            YTest = np.concatenate((YTest, nc2), axis=0)
                        if normalSubjects[x] == 2:
                            XTestF2 = np.concatenate((XTestF2, n3F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n3C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n3F4), axis=0)
                            YTest = np.concatenate((YTest, nc3), axis=0)
                        if normalSubjects[x] == 3:
                            XTestF2 = np.concatenate((XTestF2, n4F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n4C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n4F4), axis=0)
                            YTest = np.concatenate((YTest, nc4), axis=0)
                        if normalSubjects[x] == 4:
                            XTestF2 = np.concatenate((XTestF2, n5F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n5C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n5F4), axis=0)
                            YTest = np.concatenate((YTest, nc5), axis=0)
                        if normalSubjects[x] == 5:
                            XTestF2 = np.concatenate((XTestF2, n10F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n10C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n10F4), axis=0)
                            YTest = np.concatenate((YTest, nc10), axis=0)
                        if normalSubjects[x] == 6:
                            XTestF2 = np.concatenate((XTestF2, n11F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n11C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n11F4), axis=0)
                            YTest = np.concatenate((YTest, nc11), axis=0)
                        if normalSubjects[x] == 7:
                            XTestF2 = np.concatenate((XTestF2, n16F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n16C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n16F4), axis=0)
                            YTest = np.concatenate((YTest, nc16), axis=0)
                        if normalSubjects[x] == 8:
                            XTestF2 = np.concatenate((XTestF2, nfle1F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle1C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle1F4), axis=0)
                            YTest = np.concatenate((YTest, nflec1), axis=0)
                        if normalSubjects[x] == 9:
                            XTestF2 = np.concatenate((XTestF2, nfle2F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle2C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle2F4), axis=0)
                            YTest = np.concatenate((YTest, nflec2), axis=0)
                        if normalSubjects[x] == 10:
                            XTestF2 = np.concatenate((XTestF2, nfle4F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle4C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle4F4), axis=0)
                            YTest = np.concatenate((YTest, nflec4), axis=0)
                        if normalSubjects[x] == 11:
                            XTestF2 = np.concatenate((XTestF2, nfle7F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle7C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle7F4), axis=0)
                            YTest = np.concatenate((YTest, nflec7), axis=0)
                        if normalSubjects[x] == 12:
                            XTestF2 = np.concatenate((XTestF2, nfle12F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle12C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle12F4), axis=0)
                            YTest = np.concatenate((YTest, nflec12), axis=0)
                        if normalSubjects[x] == 13:
                            XTestF2 = np.concatenate((XTestF2, nfle13F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle13C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle13F4), axis=0)
                            YTest = np.concatenate((YTest, nflec13), axis=0)
                        if normalSubjects[x] == 14:
                            XTestF2 = np.concatenate((XTestF2, nfle14F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle14C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle14F4), axis=0)
                            YTest = np.concatenate((YTest, nflec14), axis=0)
                        if normalSubjects[x] == 15:
                            XTestF2 = np.concatenate((XTestF2, nfle15F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle15C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle15F4), axis=0)
                            YTest = np.concatenate((YTest, nflec15), axis=0)

                XTrainF2 = XTrainF2.reshape(round(len(XTrainF2) / 100), 100)
                XTrainC4 = XTrainC4.reshape(round(len(XTrainC4) / 100), 100)
                XTrainF4 = XTrainF4.reshape(round(len(XTrainF4) / 100), 100)
                XTestF2 = XTestF2.reshape(round(len(XTestF2) / 100), 100)
                XTestC4 = XTestC4.reshape(round(len(XTestC4) / 100), 100)
                XTestF4 = XTestF4.reshape(round(len(XTestF4) / 100), 100)

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

                from sklearn.utils import class_weight
                class_weights = class_weight.compute_class_weight('balanced', np.unique(YTrain), YTrain)

                ################################ A phase Classification

                # expand dimensions
                if NumberOfTimeSteps == 10:  # choose either 5, 15 or 25
                    timeSteps = 10

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

                    # LSTM

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

                # NumberOfCannels = 1 -> F2
                # NumberOfCannels = 2 -> C4
                # NumberOfCannels = 3 -> F4
                # NumberOfCannels = 4 -> F2 and C4
                # NumberOfCannels = 5 -> F2 and F4
                # NumberOfCannels = 6 -> C4 and F4
                # NumberOfCannels = 7 -> F2 and C4 and F4
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

                class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
                    a = 1

                #              def on_train_batch_end(self, batch, logs=None):
                #                print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
                #
                #              def on_test_batch_end(self, batch, logs=None):
                #                print('For batch {}, loss is {:7.2f}.'.format(batch, logs['loss']))
                #
                #              def on_epoch_end(self, epoch, logs=None):
                #                print('The average loss for epoch {} is {:7.2f}.'.format(epoch, logs['loss']))

                class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
                    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

                  Arguments:
                      patience: Number of epochs to wait after min has been hit. After this
                      number of no improvement, training stops.
                  """

                    def __init__(self, patience=patienteceValue):
                        super(EarlyStoppingAtMinLoss, self).__init__()

                        self.patience = patience

                        # best_weights to store the weights at which the minimum loss occurs.
                        self.best_weights = None

                    def on_train_begin(self, logs=None):
                        # The number of epoch it has waited when loss is no longer minimum.
                        self.wait = 0
                        # The epoch the training stops at.
                        self.stopped_epoch = 0
                        # Initialize the best as infinity.
                        #    self.best = np.Inf
                        self.best = 0.2
                        self._data = []
                        self.curentAUC = 0.2
                        print('Train started')

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
                        current = auc_keras

                        print('AUC : ', current)

                        if np.greater(self.curentAUC, self.best + 0.005):
                            print('Update')
                            self.best = self.curentAUC
                            self.wait = 0
                            # Record the best weights if current results is better (less)
                            self.best_weights = self.model.get_weights()
                        else:
                            self.wait += 1
                            if self.wait >= self.patience:
                                self.stopped_epoch = epoch
                                self.model.stop_training = True
                                print('Restoring model weights from the end of the best epoch.')
                                self.model.set_weights(self.best_weights)

                    def on_train_end(self, logs=None):
                        if self.stopped_epoch > 0:
                            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['binary_accuracy'])

                # NumberOfCannels = 1 -> F2
                # NumberOfCannels = 2 -> C4
                # NumberOfCannels = 3 -> F4
                # NumberOfCannels = 4 -> F2 and C4
                # NumberOfCannels = 5 -> F2 and F4
                # NumberOfCannels = 6 -> C4 and F4
                # NumberOfCannels = 7 -> F2 and C4 and F4
                if NumberOfCannels == 1:
                    x_train, x_valid, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
                    model.fit(x_train, y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=(x_valid, y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict(XTestF2)
                elif NumberOfCannels == 2:
                    x_train, x_valid, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
                    model.fit(x_train, y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=(x_valid, y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict(XTestC4)
                elif NumberOfCannels == 3:
                    x_train, x_valid, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
                    model.fit(x_train, y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=(x_valid, y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict(XTestF4)
                elif NumberOfCannels == 4:
                    x_trainF2, x_validF2, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
                    x_trainC4, x_validC4, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
                    model.fit([x_trainF2, x_trainC4], y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=([x_validF2, x_validC4], y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict([XTestF2, XTestC4])
                elif NumberOfCannels == 5:
                    x_trainF2, x_validF2, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
                    x_trainF4, x_validF4, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
                    model.fit([x_trainF2, x_trainF4], y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=([x_validF2, x_validF4], y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict([XTestF2, XTestF4])
                elif NumberOfCannels == 6:
                    x_trainC4, x_validC4, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
                    x_trainF4, x_validF4, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
                    model.fit([x_trainC4, x_trainF4], y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=([x_validC4, x_validF4], y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict([XTestC4, XTestF4])
                else:
                    x_trainF2, x_validF2, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
                    x_trainC4, x_validC4, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
                    x_trainF4, x_validF4, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
                    model.fit([x_trainF2, x_trainC4, x_trainF4], y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=([x_validF2, x_validC4, x_validF4], y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict([XTestF2, XTestC4, XTestF4])

                YTestOneLine = np.zeros(len(YTest))
                for x in range(len(YTest)):
                    if YTest[x, 0] == 1:
                        YTestOneLine[x] = 0
                    else:
                        YTestOneLine[x] = 1

                predictiony_predMean = np.mean(proba[:, 0])
                predictiony_pred = np.zeros(len(YTestOneLine))
                for x in range(len(YTestOneLine)):
                    if proba[x, 0] > predictiony_predMean:
                        predictiony_pred[x] = 0
                    else:
                        predictiony_pred[x] = 1

                tn, fp, fn, tp = confusion_matrix(YTestOneLine, predictiony_pred).ravel()
                print(classification_report(YTestOneLine, predictiony_pred))
                accuracy0 = (tp + tn) / (tp + tn + fp + fn)
                print('Accuracy : ', accuracy0)
                sensitivity0 = tp / (tp + fn)
                print('Sensitivity : ', sensitivity0)
                specificity0 = tn / (fp + tn)
                print('Specificity : ', specificity0)

                fpr_keras, tpr_keras, thresholds_keras = roc_curve(YTestOneLine, proba[:, 1])
                auc_keras = auc(fpr_keras, tpr_keras)
                print('AUC : ', auc_keras)

                capPredictedPredicted = predictiony_pred
                for k in range(len(capPredictedPredicted) - 1):
                    if k > 0:
                        if capPredictedPredicted[k - 1] == 0 and capPredictedPredicted[k] == 1 and capPredictedPredicted[k + 1] == 0:
                            capPredictedPredicted[k] = 0

                for k in range(len(capPredictedPredicted) - 1):
                    if k > 0:
                        if capPredictedPredicted[k - 1] == 1 and capPredictedPredicted[k] == 0 and capPredictedPredicted[k + 1] == 1:
                            capPredictedPredicted[k] = 1

                tn, fp, fn, tp = confusion_matrix(YTestOneLine, capPredictedPredicted).ravel()
                print(classification_report(YTestOneLine, capPredictedPredicted))
                accuracy0 = (tp + tn) / (tp + tn + fp + fn)
                print('Accuracy : ', accuracy0)
                sensitivity0 = tp / (tp + fn)
                print('Sensitivity : ', sensitivity0)
                specificity0 = tn / (fp + tn)
                print('Specificity : ', specificity0)
                AccAtEnd[ee] = accuracy0
                SenAtEnd[ee] = sensitivity0
                SpeAtEnd[ee] = specificity0
                AUCAtEnd[ee] = auc_keras

                TPEnd[ee] = tp
                TNEnd[ee] = tn
                FPEnd[ee] = fp
                FNEnd[ee] = fn

                del XTrainF2, XTrainC4, XTrainF4, YTrain, XTestF2, XTestC4, XTestF4, YTest, model

            #####################################################################################################################################

            else:  ## second fold
                gc.collect()
                tf.keras.backend.clear_session()

                if normalSubjects[Begin] == 0:
                    XTestF2 = n1F2
                    XTestC4 = n1C4
                    XTestF4 = n1F4
                    YTest = nc1
                elif normalSubjects[Begin] == 1:
                    XTestF2 = n2F2
                    XTestC4 = n2C4
                    XTestF4 = n2F4
                    YTest = nc2
                elif normalSubjects[Begin] == 2:
                    XTestF2 = n3F2
                    XTestC4 = n3C4
                    XTestF4 = n3F4
                    YTest = nc3
                elif normalSubjects[Begin] == 3:
                    XTestF2 = n4F2
                    XTestC4 = n4C4
                    XTestF4 = n4F4
                    YTest = nc4
                elif normalSubjects[Begin] == 4:
                    XTestF2 = n5F2
                    XTestC4 = n5C4
                    XTestF4 = n5F4
                    YTest = nc5
                elif normalSubjects[Begin] == 5:
                    XTestF2 = n10F2
                    XTestC4 = n10C4
                    XTestF4 = n10F4
                    YTest = nc10
                elif normalSubjects[Begin] == 6:
                    XTestF2 = n11F2
                    XTestC4 = n11C4
                    XTestF4 = n11F4
                    YTest = nc11
                elif normalSubjects[Begin] == 7:
                    XTestF2 = n16F2
                    XTestC4 = n16C4
                    XTestF4 = n16F4
                    YTest = nc16
                elif normalSubjects[Begin] == 8:
                    XTestF2 = nfle1F2
                    XTestC4 = nfle1C4
                    XTestF4 = nfle1F4
                    YTest = nflec1
                elif normalSubjects[Begin] == 9:
                    XTestF2 = nfle2F2
                    XTestC4 = nfle2C4
                    XTestF4 = nfle2F4
                    YTest = nflec2
                elif normalSubjects[Begin] == 10:
                    XTestF2 = nfle4F2
                    XTestC4 = nfle4C4
                    XTestF4 = nfle4F4
                    YTest = nflec4
                elif normalSubjects[Begin] == 11:
                    XTestF2 = nfle7F2
                    XTestC4 = nfle7C4
                    XTestF4 = nfle7F4
                    YTest = nflec7
                elif normalSubjects[Begin] == 12:
                    XTestF2 = nfle12F2
                    XTestC4 = nfle12C4
                    XTestF4 = nfle12F4
                    YTest = nflec12
                elif normalSubjects[Begin] == 13:
                    XTestF2 = nfle13F2
                    XTestC4 = nfle13C4
                    XTestF4 = nfle13F4
                    YTest = nflec13
                elif normalSubjects[Begin] == 14:
                    XTestF2 = nfle14F2
                    XTestC4 = nfle14C4
                    XTestF4 = nfle14F4
                    YTest = nflec14
                else:
                    XTestF2 = nfle15F2
                    XTestC4 = nfle15C4
                    XTestF4 = nfle15F4
                    YTest = nflec15

                if normalSubjects[BeginTest] == 0:
                    XTrainF2 = n1F2
                    XTrainC4 = n1C4
                    XTrainF4 = n1F4
                    YTrain = nc1
                elif normalSubjects[BeginTest] == 1:
                    XTrainF2 = n2F2
                    XTrainC4 = n2C4
                    XTrainF4 = n2F4
                    YTrain = nc2
                elif normalSubjects[BeginTest] == 2:
                    XTrainF2 = n3F2
                    XTrainC4 = n3C4
                    XTrainF4 = n3F4
                    YTrain = nc3
                elif normalSubjects[BeginTest] == 3:
                    XTrainF2 = n4F2
                    XTrainC4 = n4C4
                    XTrainF4 = n4F4
                    YTrain = nc4
                elif normalSubjects[BeginTest] == 4:
                    XTrainF2 = n5F2
                    XTrainC4 = n5C4
                    XTrainF4 = n5F4
                    YTrain = nc5
                elif normalSubjects[BeginTest] == 5:
                    XTrainF2 = n10F2
                    XTrainC4 = n10C4
                    XTrainF4 = n10F4
                    YTrain = nc10
                elif normalSubjects[BeginTest] == 6:
                    XTrainF2 = n11F2
                    XTrainC4 = n11C4
                    XTrainF4 = n11F4
                    YTrain = nc11
                elif normalSubjects[BeginTest] == 7:
                    XTrainF2 = n16F2
                    XTrainC4 = n16C4
                    XTrainF4 = n16F4
                    YTrain = nc16
                elif normalSubjects[BeginTest] == 8:
                    XTrainF2 = nfle1F2
                    XTrainC4 = nfle1C4
                    XTrainF4 = nfle1F4
                    YTrain = nflec1
                elif normalSubjects[BeginTest] == 9:
                    XTrainF2 = nfle2F2
                    XTrainC4 = nfle2C4
                    XTrainF4 = nfle2F4
                    YTrain = nflec2
                elif normalSubjects[BeginTest] == 10:
                    XTrainF2 = nfle4F2
                    XTrainC4 = nfle4C4
                    XTrainF4 = nfle4F4
                    YTrain = nflec4
                elif normalSubjects[BeginTest] == 11:
                    XTrainF2 = nfle7F2
                    XTrainC4 = nfle7C4
                    XTrainF4 = nfle7F4
                    YTrain = nflec7
                elif normalSubjects[BeginTest] == 12:
                    XTrainF2 = nfle12F2
                    XTrainC4 = nfle12C4
                    XTrainF4 = nfle12F4
                    YTrain = nflec12
                elif normalSubjects[BeginTest] == 13:
                    XTrainF2 = nfle13F2
                    XTrainC4 = nfle13C4
                    XTrainF4 = nfle13F4
                    YTrain = nflec13
                elif normalSubjects[BeginTest] == 14:
                    XTrainF2 = nfle14F2
                    XTrainC4 = nfle14C4
                    XTrainF4 = nfle14F4
                    YTrain = nflec14
                else:
                    XTrainF2 = nfle15F2
                    XTrainC4 = nfle15C4
                    XTrainF4 = nfle15F4
                    YTrain = nflec15

                for x in range(20):
                    if x < BeginTest and x > Begin:
                        if normalSubjects[x] == 0:
                            XTestF2 = np.concatenate((XTestF2, n1F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n1C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n1F4), axis=0)
                            YTest = np.concatenate((YTest, nc1), axis=0)
                        if normalSubjects[x] == 1:
                            XTestF2 = np.concatenate((XTestF2, n2F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n2C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n2F4), axis=0)
                            YTest = np.concatenate((YTest, nc2), axis=0)
                        if normalSubjects[x] == 2:
                            XTestF2 = np.concatenate((XTestF2, n3F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n3C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n3F4), axis=0)
                            YTest = np.concatenate((YTest, nc3), axis=0)
                        if normalSubjects[x] == 3:
                            XTestF2 = np.concatenate((XTestF2, n4F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n4C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n4F4), axis=0)
                            YTest = np.concatenate((YTest, nc4), axis=0)
                        if normalSubjects[x] == 4:
                            XTestF2 = np.concatenate((XTestF2, n5F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n5C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n5F4), axis=0)
                            YTest = np.concatenate((YTest, nc5), axis=0)
                        if normalSubjects[x] == 5:
                            XTestF2 = np.concatenate((XTestF2, n10F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n10C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n10F4), axis=0)
                            YTest = np.concatenate((YTest, nc10), axis=0)
                        if normalSubjects[x] == 6:
                            XTestF2 = np.concatenate((XTestF2, n11F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n11C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n11F4), axis=0)
                            YTest = np.concatenate((YTest, nc11), axis=0)
                        if normalSubjects[x] == 7:
                            XTestF2 = np.concatenate((XTestF2, n16F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, n16C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, n16F4), axis=0)
                            YTest = np.concatenate((YTest, nc16), axis=0)
                        if normalSubjects[x] == 8:
                            XTestF2 = np.concatenate((XTestF2, nfle1F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle1C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle1F4), axis=0)
                            YTest = np.concatenate((YTest, nflec1), axis=0)
                        if normalSubjects[x] == 9:
                            XTestF2 = np.concatenate((XTestF2, nfle2F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle2C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle2F4), axis=0)
                            YTest = np.concatenate((YTest, nflec2), axis=0)
                        if normalSubjects[x] == 10:
                            XTestF2 = np.concatenate((XTestF2, nfle4F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle4C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle4F4), axis=0)
                            YTest = np.concatenate((YTest, nflec4), axis=0)
                        if normalSubjects[x] == 11:
                            XTestF2 = np.concatenate((XTestF2, nfle7F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle7C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle7F4), axis=0)
                            YTest = np.concatenate((YTest, nflec7), axis=0)
                        if normalSubjects[x] == 12:
                            XTestF2 = np.concatenate((XTestF2, nfle12F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle12C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle12F4), axis=0)
                            YTest = np.concatenate((YTest, nflec12), axis=0)
                        if normalSubjects[x] == 13:
                            XTestF2 = np.concatenate((XTestF2, nfle13F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle13C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle13F4), axis=0)
                            YTest = np.concatenate((YTest, nflec13), axis=0)
                        if normalSubjects[x] == 14:
                            XTestF2 = np.concatenate((XTestF2, nfle14F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle14C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle14F4), axis=0)
                            YTest = np.concatenate((YTest, nflec14), axis=0)
                        if normalSubjects[x] == 15:
                            XTestF2 = np.concatenate((XTestF2, nfle15F2), axis=0)
                            XTestC4 = np.concatenate((XTestC4, nfle15C4), axis=0)
                            XTestF4 = np.concatenate((XTestF4, nfle15F4), axis=0)
                            YTest = np.concatenate((YTest, nflec15), axis=0)

                    if x <= numberSubjects - 1 and x >= BeginTest:
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

                for i in range(0, len(YTrain), 1):
                    if YTrain[i] > 0:
                        YTrain[i] = 1
                    else:
                        YTrain[i] = 0
                for i in range(0, len(YTest), 1):
                    if YTest[i] > 0:
                        YTest[i] = 1
                    else:
                        YTest[i] = 0

                from sklearn.utils import class_weight
                class_weights = class_weight.compute_class_weight('balanced', np.unique(YTrain), YTrain)

                if NumberOfTimeSteps == 10:
                    timeSteps = 10

                    for ii in range(timeSteps - 1):
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

                    for ii in range(timeSteps - 1):
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

                    for ii in range(timeSteps - 1):
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

                    for ii in range(timeSteps - 1):
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

                    # LSTM

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

                class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
                    a = 1

                class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):

                    def __init__(self, patience=patienteceValue):
                        super(EarlyStoppingAtMinLoss, self).__init__()

                        self.patience = patience
                        self.best_weights = None

                    def on_train_begin(self, logs=None):
                        self.wait = 0
                        self.stopped_epoch = 0
                        self.best = 0.2
                        self._data = []
                        self.curentAUC = 0.2
                        print('Train started')

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
                        current = auc_keras
                        print('AUC : ', current)

                        if np.greater(self.curentAUC, self.best + 0.005):
                            print('Update')
                            self.best = self.curentAUC
                            self.wait = 0
                            self.best_weights = self.model.get_weights()
                        else:
                            self.wait += 1
                            if self.wait >= self.patience:
                                self.stopped_epoch = epoch
                                self.model.stop_training = True
                                print('Restoring model weights from the end of the best epoch.')
                                self.model.set_weights(self.best_weights)

                    def on_train_end(self, logs=None):
                        if self.stopped_epoch > 0:
                            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

                model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['binary_accuracy'])

                if NumberOfCannels == 1:
                    x_train, x_valid, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
                    model.fit(x_train, y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=(x_valid, y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict(XTestF2)
                elif NumberOfCannels == 2:
                    x_train, x_valid, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
                    model.fit(x_train, y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=(x_valid, y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict(XTestC4)
                elif NumberOfCannels == 3:
                    x_train, x_valid, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
                    model.fit(x_train, y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=(x_valid, y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict(XTestF4)
                elif NumberOfCannels == 4:
                    x_trainF2, x_validF2, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
                    x_trainC4, x_validC4, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
                    model.fit([x_trainF2, x_trainC4], y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=([x_validF2, x_validC4], y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict([XTestF2, XTestC4])
                elif NumberOfCannels == 5:
                    x_trainF2, x_validF2, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
                    x_trainF4, x_validF4, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
                    model.fit([x_trainF2, x_trainF4], y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=([x_validF2, x_validF4], y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict([XTestF2, XTestF4])
                elif NumberOfCannels == 6:
                    x_trainC4, x_validC4, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
                    x_trainF4, x_validF4, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
                    model.fit([x_trainC4, x_trainF4], y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=([x_validC4, x_validF4], y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict([XTestC4, XTestF4])
                else:
                    x_trainF2, x_validF2, y_train, y_valid = train_test_split(XTrainF2, YTrain, test_size=0.33, shuffle=False)
                    x_trainC4, x_validC4, y_train, y_valid = train_test_split(XTrainC4, YTrain, test_size=0.33, shuffle=False)
                    x_trainF4, x_validF4, y_train, y_valid = train_test_split(XTrainF4, YTrain, test_size=0.33, shuffle=False)
                    model.fit([x_trainF2, x_trainC4, x_trainF4], y_train,
                              batch_size=BatchSize,
                              epochs=20,
                              validation_data=([x_validF2, x_validC4, x_validF4], y_valid),
                              verbose=1,
                              class_weight=class_weights, callbacks=[LossAndErrorPrintingCallback(), EarlyStoppingAtMinLoss()])
                    print("Testing epoch", ee)
                    proba = model.predict([XTestF2, XTestC4, XTestF4])

                YTestOneLine = np.zeros(len(YTest))
                for x in range(len(YTest)):
                    if YTest[x, 0] == 1:
                        YTestOneLine[x] = 0
                    else:
                        YTestOneLine[x] = 1

                predictiony_predMean = np.mean(proba[:, 0])
                predictiony_pred = np.zeros(len(YTestOneLine))
                for x in range(len(YTestOneLine)):
                    if proba[x, 0] > predictiony_predMean:
                        predictiony_pred[x] = 0
                    else:
                        predictiony_pred[x] = 1

                tn, fp, fn, tp = confusion_matrix(YTestOneLine, predictiony_pred).ravel()
                print(classification_report(YTestOneLine, predictiony_pred))
                accuracy0 = (tp + tn) / (tp + tn + fp + fn)
                print('Accuracy : ', accuracy0)
                sensitivity0 = tp / (tp + fn)
                print('Sensitivity : ', sensitivity0)
                specificity0 = tn / (fp + tn)
                print('Specificity : ', specificity0)

                fpr_keras, tpr_keras, thresholds_keras = roc_curve(YTestOneLine, proba[:, 1])
                auc_keras = auc(fpr_keras, tpr_keras)
                print('AUC : ', auc_keras)

                capPredictedPredicted = predictiony_pred
                for k in range(len(capPredictedPredicted) - 1):
                    if k > 0:
                        if capPredictedPredicted[k - 1] == 0 and capPredictedPredicted[k] == 1 and capPredictedPredicted[k + 1] == 0:
                            capPredictedPredicted[k] = 0

                for k in range(len(capPredictedPredicted) - 1):
                    if k > 0:
                        if capPredictedPredicted[k - 1] == 1 and capPredictedPredicted[k] == 0 and capPredictedPredicted[k + 1] == 1:
                            capPredictedPredicted[k] = 1

                tn, fp, fn, tp = confusion_matrix(YTestOneLine, capPredictedPredicted).ravel()
                print(classification_report(YTestOneLine, capPredictedPredicted))
                accuracy0 = (tp + tn) / (tp + tn + fp + fn)
                print('Accuracy : ', accuracy0)
                sensitivity0 = tp / (tp + fn)
                print('Sensitivity : ', sensitivity0)
                specificity0 = tn / (fp + tn)
                print('Specificity : ', specificity0)
                AccAtEnd[ee] = accuracy0
                SenAtEnd[ee] = sensitivity0
                SpeAtEnd[ee] = specificity0
                AUCAtEnd[ee] = auc_keras

                TPEnd[ee] = tp
                TNEnd[ee] = tn
                FPEnd[ee] = fp
                FNEnd[ee] = fn

                del XTrainF2, XTrainC4, XTrainF4, YTrain, XTestF2, XTestC4, XTestF4, YTest, model

        print('Final results: ')

        print('Ave Accuracy : ', np.mean(AccAtEnd) * 100)
        print('Ave Sensitivity : ', np.mean(SenAtEnd) * 100)
        print('Ave Specificity : ', np.mean(SpeAtEnd) * 100)
        print('Ave AUC : ', np.mean(AUCAtEnd) * 100)

        print('STD Accuracy : ', np.std(AccAtEnd) * 100)
        print('STD Sensitivity : ', np.std(SenAtEnd) * 100)
        print('STD Specificity : ', np.std(SpeAtEnd) * 100)
        print('STD AUC : ', np.std(AUCAtEnd) * 100)

        if indicationsGen == 0:
            f = open("AccMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.mean(AccAtEnd) * 100, f)
            f.close()
            f = open("SenMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.mean(SenAtEnd) * 100, f)
            f.close()
            f = open("SpeMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.mean(SpeAtEnd) * 100, f)
            f.close()
            f = open("AUCMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.mean(AUCAtEnd) * 100, f)
            f.close()
            f = open("AccStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.std(AccAtEnd) * 100, f)
            f.close()
            f = open("SenStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.std(SenAtEnd) * 100, f)
            f.close()
            f = open("SpeStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.std(SpeAtEnd) * 100, f)
            f.close()
            f = open("AUCStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.std(AUCAtEnd) * 100, f)
            f.close()
            f = open("TPMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.mean(TPEnd) * 100, f)
            f.close()
            f = open("TNMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.mean(TNEnd) * 100, f)
            f.close()
            f = open("FPMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.mean(FPEnd) * 100, f)
            f.close()
            f = open("FNMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.mean(FNEnd) * 100, f)
            f.close()
            f = open("TPStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.std(TPEnd) * 100, f)
            f.close()
            f = open("TNStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.std(TNEnd) * 100, f)
            f.close()
            f = open("FPStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.std(FPEnd) * 100, f)
            f.close()
            f = open("FNStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump("New generation: ", f)
            pickle.dump(np.std(FNEnd) * 100, f)
            f.close()
        else:
            f = open("AccMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.mean(AccAtEnd) * 100, f)
            f.close()
            f = open("SenMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.mean(SenAtEnd) * 100, f)
            f.close()
            f = open("SpeMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.mean(SpeAtEnd) * 100, f)
            f.close()
            f = open("AUCMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.mean(AUCAtEnd) * 100, f)
            f.close()
            f = open("AccStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.std(AccAtEnd) * 100, f)
            f.close()
            f = open("SenStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.std(SenAtEnd) * 100, f)
            f.close()
            f = open("SpeStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.std(SpeAtEnd) * 100, f)
            f.close()
            f = open("AUCStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.std(AUCAtEnd) * 100, f)
            f.close()
            f = open("TPMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.mean(TPEnd) * 100, f)
            f.close()
            f = open("TNMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.mean(TNEnd) * 100, f)
            f.close()
            f = open("FPMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.mean(FPEnd) * 100, f)
            f.close()
            f = open("FNMean " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.mean(FNEnd) * 100, f)
            f.close()
            f = open("TPStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.std(TPEnd) * 100, f)
            f.close()
            f = open("TNStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.std(TNEnd) * 100, f)
            f.close()
            f = open("FPStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.std(FPEnd) * 100, f)
            f.close()
            f = open("FNStd " + str(GenerationsFile) + ".txt", 'ab')
            pickle.dump(np.std(FNEnd) * 100, f)
            f.close()

        scores.append(np.mean(AUCAtEnd) * 100)
        indicationsGen = 1

    scores, population = np.array(scores), np.array(population)

    inds = np.flip(np.argsort(scores))
    scoresGenTemp = np.zeros((NumbParents))
    population_nextgenTemp = np.zeros((NumbParents, NumbBitsDecimal))
    for sortingLines in range(0, len(inds), 1):
        scoresGenTemp[sortingLines] = scores[inds[sortingLines]]
        population_nextgenTemp[sortingLines, :] = population[inds[sortingLines], :]

    return scoresGenTemp, population_nextgenTemp


def selection(pop_after_fit, n_parents):
    """
    Method for select the population
    """
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel, corss_Rate):
    """
    Method for creating the crossover
    """
    population_nextgen_crossover = []
    for i in range(0, len(pop_after_sel)):
        if random.random() <= corss_Rate:
            while True:
                numbers1 = random.randrange(0, len(pop_after_sel))
                numbers2 = random.randrange(0, len(pop_after_sel))
                parent1 = min(numbers1, numbers2)
                numbers1 = random.randrange(0, len(pop_after_sel))
                numbers2 = random.randrange(0, len(pop_after_sel))
                parent2 = min(numbers1, numbers2)
                if parent1 != parent2:
                    break
            CorssoverPoints1 = random.randrange(0, NumbBitsDecimal) + 1
            CorssoverPoints2 = random.randrange(0, NumbBitsDecimal) + 1
            CorssoverPoints = np.sort([CorssoverPoints1, CorssoverPoints2])
            child = pop_after_sel[parent1]
            child[CorssoverPoints[0] + 1:CorssoverPoints[1]] = pop_after_sel[parent2][CorssoverPoints[0] + 1:CorssoverPoints[1]]
        else:
            numbers1 = random.randrange(0, len(pop_after_sel))
            numbers2 = random.randrange(0, len(pop_after_sel))
            parent1 = min(numbers1, numbers2)
            child = pop_after_sel[parent1]
        population_nextgen_crossover.append(child)
    return population_nextgen_crossover


def mutation(pop_after_cross, mutation_rate, gen):
    """
    Method for making mutation on the population
    """
    mutation_rate = mutationRate - mutationRate * int(gen / 5) * 0.3
    if mutation_rate < 0.01:
        mutation_rate = 0.01

    population_nextgen_mutation = []
    for i in range(0, len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        chromosome1 = np.zeros(len(chromosome))
        for j in range(0, len(chromosome)):
            if random.random() <= mutation_rate:
                chromosome1[j] = not chromosome[j]
            else:
                chromosome1[j] = chromosome[j]
        population_nextgen_mutation.append(chromosome1)
    return population_nextgen_mutation


def generations(NumbParents, mutation_rate, corssover_rate, gen, population_nextgen, scoresGen, elitsNumber):
    """
    Method for creating the next generation
    """
    parents = np.zeros((NumbParents, NumbBitsDecimal))
    for LinesAppendParents in range(0, NumbParents, 1):
        parents[LinesAppendParents, :] = population_nextgen[LinesAppendParents, :].copy()
    score_parents = scoresGen.copy()

    f = open("PopulationParentsBegin " + str(gen) + ".txt", 'ab')
    pickle.dump(parents, f)
    f.close()

    f = open("ScoreParentsBegin " + str(gen) + ".txt", 'ab')
    pickle.dump(score_parents, f)
    f.close()

    pop_after_cross = crossover(parents, corssover_rate)

    objectsCh = []
    with (open("PopulationParentsBegin " + str(gen) + ".txt", "rb")) as openfile:
        while True:
            try:
                objectsCh.append(pickle.load(openfile))
            except EOFError:
                break
    parents = np.asarray(objectsCh[0])

    f = open("ChildrenAfterCrossOver " + str(gen) + ".txt", 'ab')
    pickle.dump(pop_after_cross, f)
    f.close()

    population_Child = mutation(pop_after_cross, mutation_rate, gen)

    objectsCh = []
    with (open("PopulationParentsBegin " + str(gen) + ".txt", "rb")) as openfile:
        while True:
            try:
                objectsCh.append(pickle.load(openfile))
            except EOFError:
                break
    parents = np.asarray(objectsCh[0])

    f = open("ChildrenAfterMutation " + str(gen) + ".txt", 'ab')
    pickle.dump(population_Child, f)
    f.close()

    scoresChild, pop_after_fitChild = fitness_score(population_Child, gen)

    pop_after_fitChild = np.asarray(pop_after_fitChild)
    scoresChild = np.asarray(scoresChild)

    f = open("ChildrenAfterSorting " + str(gen) + ".txt", 'ab')
    pickle.dump(pop_after_fitChild, f)
    f.close()
    f = open("ScoreChildrenAfterSorting " + str(gen) + ".txt", 'ab')
    pickle.dump(scoresChild, f)
    f.close()

    tf.keras.backend.clear_session()

    population_Final = np.zeros(((NumbParents * 2) - elitsNumber, NumbBitsDecimal))
    for LinesAppend in range(0, (NumbParents * 2) - elitsNumber, 1):
        if LinesAppend < NumbParents - elitsNumber:
            population_Final[LinesAppend, :] = parents[LinesAppend + elitsNumber, :].copy()
        else:
            population_Final[LinesAppend, :] = pop_after_fitChild[LinesAppend - NumbParents + elitsNumber, :].copy()

    scoresGenFinal = np.append(score_parents[elitsNumber:], scoresChild, axis=0)

    f = open("ParentsAndChildren " + str(gen) + ".txt", 'ab')
    pickle.dump(population_Final, f)
    f.close()
    f = open("ScoreParentsAndChildren " + str(gen) + ".txt", 'ab')
    pickle.dump(scoresGenFinal, f)
    f.close()

    inds = np.flip(np.argsort(scoresGenFinal))

    scoresGenTemp = np.zeros(((NumbParents * 2) - elitsNumber))
    population_nextgenTemp = np.zeros(((NumbParents * 2) - elitsNumber, NumbBitsDecimal))
    for sortingLines in range(0, len(inds), 1):
        scoresGenTemp[sortingLines] = scoresGenFinal[inds[sortingLines]].copy()
        population_nextgenTemp[sortingLines, :] = population_Final[inds[sortingLines], :].copy()

    f = open("ParentsAndChildrenAfterSorting " + str(gen) + ".txt", 'ab')
    pickle.dump(population_nextgenTemp, f)
    f.close()
    f = open("ScoreParentsAndChildrenAfterSorting " + str(gen) + ".txt", 'ab')
    pickle.dump(scoresGenTemp, f)
    f.close()

    scoresGenUpdate = np.zeros((NumbParents))
    population_nextgenUpdate = np.zeros((NumbParents, NumbBitsDecimal))

    ################
    # Make elitism #
    ################
    for k in range(0, elitsNumber):
        scoresGenUpdate[k] = score_parents[k].copy()
        population_nextgenUpdate[k, :] = parents[k, :].copy()

    scoresGenUpdate[elitsNumber:NumbParents] = scoresGenTemp[0:NumbParents - elitsNumber].copy()
    population_nextgenUpdate[elitsNumber:NumbParents, :] = population_nextgenTemp[0:NumbParents - elitsNumber, :].copy()

    f = open("BestPopulationBeforeSorting " + str(gen) + ".txt", 'ab')
    pickle.dump(population_nextgenUpdate, f)
    f.close()
    f = open("ScoreBestPopulationBeforeSorting " + str(gen) + ".txt", 'ab')
    pickle.dump(scoresGenUpdate, f)
    f.close()

    inds = np.flip(np.argsort(scoresGenUpdate))

    scoresGenUpdateSorted = np.zeros((NumbParents))
    population_nextgenUpdateSorted = np.zeros((NumbParents, NumbBitsDecimal))
    for sortingLines in range(0, len(inds), 1):
        scoresGenUpdateSorted[sortingLines] = scoresGenUpdate[inds[sortingLines]].copy()
        population_nextgenUpdateSorted[sortingLines, :] = population_nextgenUpdate[inds[sortingLines], :].copy()

    f = open("BestPopulationAfterSorting " + str(gen) + ".txt", 'ab')
    pickle.dump(population_nextgenUpdateSorted, f)
    f.close()
    f = open("ScoreBestPopulationAfterSorting " + str(gen) + ".txt", 'ab')
    pickle.dump(scoresGenUpdateSorted, f)
    f.close()

    best_chromo.append(population_nextgenUpdateSorted[0])
    best_score.append(scoresGenUpdateSorted[0])

    f = open("Best_chromo " + str(gen) + ".txt", 'ab')
    pickle.dump(best_chromo, f)
    f.close()
    f = open("Best_score " + str(gen) + ".txt", 'ab')
    pickle.dump(best_score, f)
    f.close()

    return scoresGenUpdateSorted, population_nextgenUpdateSorted


#############################
# Control variablees for GA #
#############################
NumbParents = 15
SizePopulation = NumbParents
NumbBitsDecimal = 15
corssoverRate = 0.9
mutationRate = 0.2
NumbGenerations = 51
elitsNumber = 2

size = SizePopulation
corssover_rate = corssoverRate
n_feat = NumbBitsDecimal
n_parents = NumbParents
mutation_rate = mutationRate
n_gen = NumbGenerations

best_chromo = []
best_score = []

seconds = time.time()
population_nextgen = initilization_of_population(size, n_feat)

GenerationsFile = 0
scoresGen, pop_after_fitGen = fitness_score(population_nextgen, GenerationsFile)
tf.keras.backend.clear_session()
population_nextgen = pop_after_fitGen

f = open("Population " + str(GenerationsFile) + ".txt", 'ab')
pickle.dump("New generation: ", f)
pickle.dump(population_nextgen, f)
f.close()
f = open("Score " + str(GenerationsFile) + ".txt", 'ab')
pickle.dump("New generation: ", f)
pickle.dump(scoresGen, f)
f.close()

best_chromo.append(population_nextgen[0])
best_score.append(scoresGen[0])

secondsGen = np.zeros((n_gen))
for gen in range(1, n_gen):
    secondsGen[gen - 1] = time.time() - seconds
    seconds = time.time()
    print("\n\n")
    print("New generation: ", gen)
    print("\n\n")
    tf.keras.backend.clear_session()
    scoresGenT, population_nextgenT = generations(NumbParents, mutation_rate, corssover_rate, gen, population_nextgen, scoresGen, elitsNumber)
    f = open("Time " + str(GenerationsFile) + ".txt", 'ab')
    pickle.dump("New generation: ", f)
    pickle.dump(secondsGen, f)
    f.close()
    population_nextgen = np.copy(population_nextgenT)
    scoresGen = np.copy(scoresGenT)
