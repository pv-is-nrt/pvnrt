# ============================================================================ #
#
#   Module of machine learning functions
#   Author: Prateek Verma
#   Created on: July 26, 2021
#
# ============================================================================ #



# ============================================================================ #
#    IMPORTS
# ============================================================================ #

import os
import random
import shutil
from pathlib import Path
import pvnrt as pv

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np

import itertools
import matplotlib.patheffects as pEffects



# ============================================================================ #
#    CONSTANTS
# ============================================================================ #

VALID_PARAMS = {
    # EXPERIMENT
    'EXP_TITLE',
    'EXP_DESCRIPTION',
    # DATA
    'DATA_FILE_PATH', # full path; name and ext can be derived from this
    # MODEL
    'MODEL_SUMMARY', # string
    'MODEL_ARCH_FILE_PATH',
    'TRAINED_MODEL_FILE_PATH',
    'USE_TRAINED_MODEL',
    # COMPILE
    'LOSS_FUNCTION',
    'METRICS',
    'LEARNING_RATE',
    # FIT
    'BATCH_SIZE',
    'EPOCHS',
    'FIT_VALIDATION_SPLIT',
    'FIT_SHUFFLE',
    # PREDICT
    'PREDICT_BATCH_SIZE',
    # EVALUATE
    'TRAIN_ACCURACY_0.5',
    'VAL_ACCURACY_0.5',
    'TEST_ACCURACY_0.5',
    'TRAIN_CM_0.5',
    'VAL_CM_0.5',
    'TEST_CM_0.5',
    'THRESHOLDS', # vary thresholds params BEGIN ->
    'TRAIN_ACCURACIES',
    'VAL_ACCURACIES',
    'TEST_ACCURACIES',
    'TRAIN_TPR',
    'VAL_TPR',
    'TEST_TPR',
    'TRAIN_FPR',
    'VAL_FPR',
    'TEST_FPR',
    'TRAIN_THRESHOLD_BEST',
    'TRAIN_THRESHOLD_INDEX_BEST',
    'VAL_THRESHOLD_BEST',
    'VAL_THRESHOLD_INDEX_BEST',
    'TEST_THRESHOLD_BEST',
    'TEST_THRESHOLD_INDEX_BEST',
    'TRAIN_ACCURACY_BEST',
    'VAL_ACCURACY_BEST',
    'TEST_ACCURACY_BEST',
    'TEST_ACCURACY_CONST', # calculated using best threshold from train/val
    'TRAIN_CM_BEST',
    'VAL_CM_BEST',
    'TEST_CM_BEST',
    'TEST_CM_CONST', # calculated using best threshold from train/val
                     # vary threshold params END <-
    # CALLBACKS
    'CALLBACKS_MONITOR',
    'CALLBACKS_MODE',
    'CALLBACKS_SAVE_BEST_ONLY',
    'CALLBACKS_SAVE_WEIGHTS_ONLY',
    'CALLBACKS_SAVE_FREQ',
    'CALLBACKS_FILE_PATH',
    # IMAGES
    'IMAGE_SIZE', # tuple of (height, width)
}



# ============================================================================ #
#    SPLIT DATA
# ============================================================================ #


#    SUBFOLDER IS CLASS
# ============================================================================ #

def dataset_splitting_subFolderIsClass(
    srcPath:str, dstPath:str='', dstSubFolderName:str='splitData', clearDestination:bool=False, moveSrcFiles:bool=False, fileExtensions:list='', 
    tvtRatio:list=[7,2,1], seed:int=None, 
    softMode:bool=True, verbose:int=0
    ):

    """
    Split a folder containing files (usually images) into training, validation and test folders treating each first-level-subfolder as a class. The source needs to contain files within subfolders whose names will be adopted as the class names. Returns the list of split file-paths in the form of a dictionary with keys corresponding to class names.

    You can choose to either move or copy the files to the destination folder and can specify the ratios in a variety of ways.

    ARGUMENTS: 
      
      srcPath (string or Path, required): is the path to the directory where the source files (within subfolders) are located. All files in the source, even if they are within subdirectories will be scanned.

      dstPath (string or Path, optional): is the path to the directory where split data is intended to be saved. A subfolder with a name specified by a parameter (default name is 'splitData') is automatically created at the destination. Further, three subfolders inside it, namely 'train', 'val' and 'test' are also automatically created. Default value is an empty string, which means that the destination subfolder will be created in the current directory.

      dstSubFolderName (string, optional): is the name for the folder to be created inside the destination path, and holds the train, validation and test directories. See above for more details. Default value is 'splitData'.

      clearDestination (boolean, optional): If set to true, will delete the destination subfolder (with the default name 'splitData') and everything inside it, if the subfolder is found to exist. Default is False, if the subfolder is found to exist, then the function will quit.

      moveSrcFiles (boolean, optional): If set to true, the function will move the files instead of copying them. Default is False.

      fileExtensions (string or list, optional): If passed, only the files with the passed file extensions (lowercase and without the .) are searched for and moved/copied. A string can be passed instead of a tuple/list if only one extension needs to be specified. Filenames ending with the exact string are utilized.

      tvtRatio (float, list of floats or ints, optional): Splits the data in the provided ratios. Ratios such as 0.7, 12, [0.8], [0.8, 0.2], [0.8, 0.1, 0.1], [6, 3], [500, 100, 200] and so on... are all valid. In most cases, the function will correctly interpret the ratios and normalize them. Ratios such as [80, 10, 5, 5] are invalid (4 elements) and will cause the function to quit. It is possible to subsample the source files, i.e. utilize only a fraction of the entire dataset by providing 3 ratios whose sum is less than or equal to 1. 

      seed (number, optional): Seed used for random sampling for images. Use the same number if it is intended to get the same TVT samples repeatedly. Default is None, which means that the sampling will be non-repeatable because random function by default uses the system time as seed.

      softMode (boolean, optional): If this feature is turned on (True), then the function will return the usual list of T,V,T files for each class and not actually create/delete directories nor copy/move any files. This mode can be useful to use the function in a context where only the path names are needed. Default is False, which means that the function will actually copy/move files around.

    DEPENDS ON:
    os, random, shutil and pathlib etc. modules.

    RETURNS:
    List of file (pathlib Paths relative to script) for training, validation and test file-lists (in that order) for each class in the form of a dictionary with keys corresponding to the class names.

    EXAMPLE:
    The following code will move all files (no file extensions passed) from the 'cats' and 'dogs' subfolders found inside the source path to the (destination path = current directory > split > train > (cats, dogs); val > (cats, dogs)) folders. Because an empty string is passed by default to the destination, it means that the split data will be created (within a subfolder) in the current directory. It will delete the specified destination subfolder (split) if it already exists. The ratio provided is a single value of 0.8, which is interpreted by the function to use 80% data for training set and remaining data for the validation set.

        SRC_PATH = '\cats and dogs'
        pv.ml.dataset_splitting_subFolderIsClass(SRC_PATH, dstSubFolderName='split', clearDestination=True, moveSrcFiles=True, tvtRatio=0.8)

    """

    # create the return dictionary
    return_dict = {}

    # convert string path into a Path object
    srcPath = Path(srcPath)
    if verbose>1:
        print('Looking for files in', srcPath)

    # Use the provided destination to create TVT directories, subdir > T,V,T. I checked that if destination is an empty string then Path ignores it and goes to the next specified string argument to create the path.
    if not softMode:
        if os.path.isdir(Path(dstPath, dstSubFolderName)):
            # If the function was passed the permission to delete the destination subfolder-
            if clearDestination:
                shutil.rmtree(Path(dstPath, dstSubFolderName))
                if verbose>1:
                    print("Destination directory already exists. Deleting that and everything within.")
                # Go ahead and create the required directories
                os.mkdir(Path(dstPath, dstSubFolderName))
                os.mkdir(Path(dstPath, dstSubFolderName, 'train'))
                os.mkdir(Path(dstPath, dstSubFolderName, 'val'))
                os.mkdir(Path(dstPath, dstSubFolderName, 'test'))
                if verbose>1:
                    print('Created', Path(dstPath, dstSubFolderName), 'and train, val, test subfolders within.')
            else:
                # No need to read further into the function because we don't have a valid destination
                print('Destination directory', Path(dstPath, dstSubFolderName), 'already exists. Function was not passed the permission to delete the destination. Function exiting.')
                return None
        else:
            os.mkdir(Path(dstPath, dstSubFolderName))
            os.mkdir(Path(dstPath, dstSubFolderName, 'train'))
            os.mkdir(Path(dstPath, dstSubFolderName, 'val'))
            os.mkdir(Path(dstPath, dstSubFolderName, 'test'))
            if verbose>1:
                print('Created', Path(dstPath, dstSubFolderName), 'and train, val, test subfolders within.')       
    
    # -------------------------------
    # Taking care of the ratio input
    # -------------------------------
    
    # Convert ratio to a list if a single value is passed
    if type(tvtRatio) is not list:
        tvtRatio = [tvtRatio]
    # makes sure the ratios passed are floating point numbers
    tvtRatio = [float(item) for item in tvtRatio]
    
    # A user can pass 1, 2, 3 or more (erroneously) values in the ratio. Process accordingly.
    if len(tvtRatio)==3:
        if sum(tvtRatio)>=1:
            # Normalize them so that their sum is 1.
            tvtRatio = [item/sum(tvtRatio) for item in tvtRatio] # sum = 1.0
            if verbose>0:
                print('3 ratios provided, interpreted as', tvtRatio)
        else:
            # IMPORTANT: if sum of the ratios is less than 1, then and only then, interpret them as subsampling (not utilizing all images). In this case tvtRatio should be left unchanged.
            if verbose>0:
                print('3 ratios provided, interpreted as', tvtRatio, 'do not add up to 1. Assuming subsampling (not utilizing all images in the source) is intended, only the displayed fraction of images will be randomly sampled.')

    elif len(tvtRatio)==2:
        # this can mean two things - user does not want a test set unless the sum of the ratios is less than one. If the sum is more than 1, then normalize the set. For example, if ratio is 8:5, normalize it to [8/13:5/13] and assume that they don't want a test set.
        if sum(tvtRatio)>=1:
            # No test set will be generated
            tvtRatio = [item/sum(tvtRatio) for item in tvtRatio]
            if verbose>0:
                print('Two ratios provided, with sum greater than or equal to 1. Assuming no test set is required. Will create training and validation set in the given ratio, normalized to', tvtRatio)
        else:
            if verbose>0:
                print('Two ratios provided, with sum less than 1. Test set will also be created with the remainder items.')
    elif len(tvtRatio)==1:
        # this definitely means no test set is desired. If the element is less than 1, then it is assumed that a validation set is desired, else it is assumed that it is not.
        if tvtRatio[0]>=1:
            # set the list to [1, 0] so that no validation or test data is created.
            tvtRatio = [1, 0]
            if verbose>0:
                print('Only one ratio, greater than or equal to 1, provided. Weird input, but okay! No validation or training set will be generated.')
        else:
            # create a validation ratio (remainder), our code needs a second ratio
            tvtRatio = [tvtRatio[0], 1-tvtRatio[0]]
            if verbose>0:
                print('Only one ratio, less than 1, provided. Validation set will be generated from the remainder items.')
    else:
        print('Invalid validation ratios. Function exiting.')
        return None

    #    Return the interpreted ratio
    # -------------------------------------------------------------------- #
    return_dict['tvtRatio'] = tvtRatio

    # ------------------------
    # Setting seed
    # ------------------------
    if seed:
        random.seed(seed)

    # -----------------------------------------
    # Split the files into tvt folders
    # -----------------------------------------

    # find all the first-level subfolders; they are the classes
    #   scandir is about 20-30 times faster than other os, walk or pathlib methods.
    #   IMPORTANT: this only returns immediate subdirectories, I checked, which is what we want.
    # A condensed 'for > if' statement
    classes = [item.name for item in os.scandir(srcPath) if item.is_dir()]

    #    Return number and name of classes to return dict
    # -------------------------------------------------------------------- #
    return_dict['classes'] = classes
    

    if verbose>1:
        print('Found', len(classes), 'immediate directories in given source path. Interpreting directory names as class names. Anticipated split per class is shown below.\n')
    # create an output table
    if verbose>0:
        print(''.ljust(80,'-'))
        print('CLASS'.ljust(30)+' | '+'TRAIN'.rjust(6)+' | '+'VAL'.rjust(6)+' | '+'TEST'.rjust(6)+' | '+'SAMPLE'.rjust(6)+' | '+'TOTAL'.rjust(6))
        print(''.ljust(80,'-'))

    #    Return the output table (heading only)
    # -------------------------------------------------------------------- #
    return_dict['Table heading'] = 'CLASS'.ljust(30)+'\t'+'TRAIN'.rjust(6)+'\t'+'VAL'.rjust(6)+'\t'+'TEST'.rjust(6)+'\t'+'SAMPLE'.rjust(6)+'\t'+'TOTAL'.rjust(6)
    # create a list to hold each row of the table
    output_table = []

    # Loop through each class/subfolder
    # ----------------------------------
    for klass in classes:

        # ------------------------
        # Build list
        # ------------------------
        
        # build path for the source subfolder=class
        class_path = Path(srcPath, klass)
        
        # initiate a filename list to hold names of all files in a given class
        filename_list = []

        # IMPORTANT: subdirectories inside a class folder are not ideal, besides they may hold duplicate file names. Intentionally not using scandir, so that files deeper in the path are also returned just in case.
        for dirPath, subfolders, files in os.walk(class_path):

            # loop through all files
            # IMPORTANT! Path (dirPath, fyle) will indeed  include a subdirectory, if it exists, in the path name. That's just how os.walk works apparently.
            for fyle in files:
                # check that the files pass the extensions test, if extension list is provided. If none provided, return all files.
                if fileExtensions != '':
                    if fyle.split('.')[-1].lower() in fileExtensions:
                        filename_list.append(Path(dirPath, fyle))
                else:
                    filename_list.append(Path(dirPath, fyle))

        # build a training set list (filenames only) in a given class
        train_list = random.sample(filename_list, int(tvtRatio[0]*len(filename_list)))
        # list containing items not in training set
        notTrain_list = [item for item in filename_list if item not in train_list]
        # build a validation set list (filenames only) in a given class
        val_list = random.sample(notTrain_list, int(tvtRatio[1]*len(filename_list)))

        # build a testing set list (filenames only) in a given class
        #   IMPORTANT: Two cases arise here
        if len(tvtRatio)==3 and sum(tvtRatio)<=1:
            # list containing items not in training set
            notTrainVal_list = [item for item in notTrain_list if item not in val_list]
            test_list = random.sample(notTrainVal_list, int(tvtRatio[2]*len(filename_list)))
        else:
            # the third parameter in the ratio list will not be used. All the remaining files are assigned to test set.
            test_list = [item for item in notTrain_list if item not in val_list]

        # calculate length of sampled data
        len_sampled_list = len(train_list) + len(val_list) + len(test_list)

        # -----------------------------
        # Add klass to the return dict
        # -----------------------------
        return_dict[klass] = [train_list, val_list, test_list]

        if verbose>0:
            print(klass[0:29].ljust(30)+' | '+str(len(train_list)).rjust(6)+' | '+str(len(val_list)).rjust(6)+' | '+str(len(test_list)).rjust(6)+' | '+str(len_sampled_list).rjust(6)+' | '+str(len(filename_list)).rjust(6))

        #    Return the printed table (row by row)
        # -------------------------------------------------------------------- #
        output_table.append(klass[0:29].ljust(30)+'\t'+str(len(train_list)).rjust(6)+'\t'+str(len(val_list)).rjust(6)+'\t'+str(len(test_list)).rjust(6)+'\t'+str(len_sampled_list).rjust(6)+'\t'+str(len(filename_list)).rjust(6))
        

        # ------------------------------------
        # Copy/move files to TVT directories
        # ------------------------------------

        if not softMode: 
            # TVT dirs are already created. Now create the class directory inside of each T, V and T
            os.mkdir(Path(dstPath, dstSubFolderName, 'train', klass))
            os.mkdir(Path(dstPath, dstSubFolderName, 'val', klass))
            os.mkdir(Path(dstPath, dstSubFolderName, 'test', klass))

            # Now move/copy the files from the source folder (class subfolders) to the destination (class subfolder) within either the T, V or T folder.
            for item in train_list:
                if moveSrcFiles:
                    shutil.move(item, Path(dstPath, dstSubFolderName, 'train', klass))
                else:
                    shutil.copy(item, Path(dstPath, dstSubFolderName, 'train', klass))
            for item in val_list:
                if moveSrcFiles:
                    shutil.move(item, Path(dstPath, dstSubFolderName, 'val', klass))
                else:
                    shutil.copy(item, Path(dstPath, dstSubFolderName, 'val', klass))
            for item in test_list:
                if moveSrcFiles:
                    shutil.move(item, Path(dstPath, dstSubFolderName, 'test', klass))
                else:
                    shutil.copy(item, Path(dstPath, dstSubFolderName, 'test', klass))
        
    if verbose>0 and not softMode:
        print('\nSuccessfully completed moving/copying files.')

    #    Return the output table data
    # -------------------------------------------------------------------- #
    
    return_dict['Table data'] = output_table
    
    
    return return_dict


#    PATHS TO TVT
# ============================================================================ #

def paths_to_tvt(
    imgPaths:list, className:str, dstPath:str='', clearClassDirs:bool=False,
    tvtRatio:list=[7,2,1], seed:int=1000, 
    softMode:bool=True, verbose:int=0
    ):

    """
    Split images into training (T), validation (V) and test folders (T). Image paths need to be provided as a list and should belong to a particular class. A subfolder with provided class name will be created within the TVT folders. Returns some statistical information about processed files and a thumbnail image (TODO: reword).

    ARGUMENTS: 
      
      imgPaths (string or Path, required): is the path to the desired image files in the form of a list of paths. TODO: add support to use files in a given directory.

      className (string, required): is the name of the class. A subfolder with this name will be created within the TVT folders.

      dstPath (string or Path, optional): is the path to the directory where split data is intended to be saved. Three subfolders inside it, namely 'train', 'val' and 'test' are automatically created. Default value is an empty string, which means that the destination subfolder will be created in the current directory.

      clearDestination (boolean, optional): If set to true, will delete the destination folder and everything inside it, if the folder is found to exist. Default is False, if the subfolder is found to exist, the function will quit.

      tvtRatio (float, list of floats or ints, optional): Splits the data in the provided ratios. Ratios such as 0.7, 12, [0.8], [0.8, 0.2], [0.8, 0.1, 0.1], [6, 3], [500, 100, 200] and so on... are all valid. In most cases, the function will correctly interpret the ratios and normalize them. Ratios such as [80, 10, 5, 5] are invalid (4 elements) and will cause the function to quit. It is possible to subsample the source files, i.e. utilize only a fraction of the entire dataset by providing 3 ratios whose sum is less than or equal to 1. 

      seed (number, optional): Seed used for random sampling for images. Use the same number if it is intended to get the same TVT samples repeatedly. Default is None, which means that the sampling will be non-repeatable because random function by default uses the system time as seed.

      softMode (boolean, optional): If this feature is turned on (True), then the function will return the usual information and not actually create/delete directories nor copy/move any files. Default is True.

    RETURNS:
    TODO: add return values

    EXAMPLE:
    TODO:

    """

    # create the return dictionary
    return_dict = {}
    return_dict['Class name'] = className

    # Use the provided destination to create TVT directories > T,V,T. I checked that if destination is an empty string then Path ignores it and goes to the next specified string argument to create the path.
    if not softMode:

        if Path(dstPath).is_dir():

            # check if the TVT folders are absent
            if not Path(dstPath, 'train').is_dir():
                os.mkdir(Path(dstPath, 'train'))
            if not Path(dstPath, 'val').is_dir():
                os.mkdir(Path(dstPath, 'val'))
            if not Path(dstPath, 'test').is_dir():
                os.mkdir(Path(dstPath, 'test'))

            # Next see if the class subfolder is present within each of the TVT folders. If yes, then delete them if clearing is permitted or quit if clearing is not permitted. If not, then create them.
            if Path(dstPath, 'train', className).is_dir():
                if clearClassDirs:
                    shutil.rmtree(Path(dstPath, 'train', className))
                    # need to make the dir again
                    os.mkdir(Path(dstPath, 'train', className))
                else:
                    print('ERROR: The class subfolder is already present in the train folder. If you want to clear the class subfolder, set the clearClassDirs flag to True. Function quitting.')
                    return None
            else:
                os.mkdir(Path(dstPath, 'train', className))
            
            if Path(dstPath, 'val', className).is_dir():
                if clearClassDirs:
                    shutil.rmtree(Path(dstPath, 'val', className))
                    # need to make the dir again
                    os.mkdir(Path(dstPath, 'val', className))
                else:
                    print('ERROR: The class subfolder is already present in the val folder. If you want to clear the class subfolder, set the clearClassDirs flag to True. Function quitting.')
                    return None
            else:
                os.mkdir(Path(dstPath, 'val', className))

            if Path(dstPath, 'test', className).is_dir():
                if clearClassDirs:
                    shutil.rmtree(Path(dstPath, 'test', className))
                    # need to make the dir again
                    os.mkdir(Path(dstPath, 'test', className))
                else:
                    print('ERROR: The class subfolder is already present in the test folder. If you want to clear the class subfolder, set the clearClassDirs flag to True. Function quitting.')
                    return None
            else:
                os.mkdir(Path(dstPath, 'test', className))

        # if the destination is not a directory, then create all required subfolders within
        else:
            os.makedirs(Path(dstPath))
            os.makedirs(Path(dstPath, 'train', className))
            os.makedirs(Path(dstPath, 'val', className))
            os.makedirs(Path(dstPath, 'test', className))
            if verbose>1:
                print('Created', Path(dstPath), 'and train, val, test subfolders within.')       
    
    # -------------------------------
    # Taking care of the ratio input
    # -------------------------------
    
    # Convert ratio to a list if a single value is passed
    if type(tvtRatio) is not list:
        tvtRatio = [tvtRatio]
    # makes sure the ratios passed are floating point numbers
    tvtRatio = [float(item) for item in tvtRatio]
    
    # A user can pass 1, 2, 3 or more (erroneously) values in the ratio. Process accordingly.
    if len(tvtRatio)==3:
        if sum(tvtRatio)>=1:
            # Normalize them so that their sum is 1.
            tvtRatio = [item/sum(tvtRatio) for item in tvtRatio] # sum = 1.0
            if verbose>0:
                print('3 ratios provided, interpreted as', tvtRatio)
        else:
            # IMPORTANT: if sum of the ratios is less than 1, then and only then, interpret them as subsampling (not utilizing all images). In this case tvtRatio should be left unchanged.
            if verbose>0:
                print('3 ratios provided, interpreted as', tvtRatio, 'do not add up to 1. Assuming subsampling (not utilizing all images in the source) is intended, only the displayed fraction of images will be randomly sampled.')

    elif len(tvtRatio)==2:
        # this can mean two things - user does not want a test set unless the sum of the ratios is less than one. If the sum is more than 1, then normalize the set. For example, if ratio is 8:5, normalize it to [8/13:5/13] and assume that they don't want a test set.
        if sum(tvtRatio)>=1:
            # No test set will be generated
            tvtRatio = [item/sum(tvtRatio) for item in tvtRatio]
            if verbose>0:
                print('Two ratios provided, with sum greater than or equal to 1. Assuming no test set is required. Will create training and validation set in the given ratio, normalized to', tvtRatio)
        else:
            if verbose>0:
                print('Two ratios provided, with sum less than 1. Test set will also be created with the remainder items.')
    elif len(tvtRatio)==1:
        # this definitely means no test set is desired. If the element is less than 1, then it is assumed that a validation set is desired, else it is assumed that it is not.
        if tvtRatio[0]>=1:
            # set the list to [1, 0] so that no validation or test data is created.
            tvtRatio = [1, 0]
            if verbose>0:
                print('Only one ratio, greater than or equal to 1, provided. Weird input, but okay! No validation or training set will be generated.')
        else:
            # create a validation ratio (remainder), our code needs a second ratio
            tvtRatio = [tvtRatio[0], 1-tvtRatio[0]]
            if verbose>0:
                print('Only one ratio, less than 1, provided. Validation set will be generated from the remainder items.')
    else:
        print('Invalid validation ratios. Function exiting.')
        return None

    #    Return the interpreted ratio
    # -------------------------------------------------------------------- #
    return_dict['TVT ratio'] = tvtRatio

    # ------------------------
    # Setting seed
    # ------------------------
    if seed:
        random.seed(seed)

    # -----------------------------------------
    # Split the files into tvt folders
    # -----------------------------------------

    # build a training set list (filenames only) in a given class
    train_list = random.sample(imgPaths, int(tvtRatio[0]*len(imgPaths)))
    # list containing items not in training set
    notTrain_list = [item for item in imgPaths if item not in train_list]
    # build a validation set list (filenames only) in a given class
    val_list = random.sample(notTrain_list, int(tvtRatio[1]*len(imgPaths)))

    # build a testing set list (filenames only) in a given class
    #   IMPORTANT: Two cases arise here
    if len(tvtRatio)==3 and sum(tvtRatio)<=1:
        # list containing items not in training set
        notTrainVal_list = [item for item in notTrain_list if item not in val_list]
        test_list = random.sample(notTrainVal_list, int(tvtRatio[2]*len(imgPaths)))
    else:
        # the third parameter in the ratio list will not be used. All the remaining files are assigned to test set.
        test_list = [item for item in notTrain_list if item not in val_list]

    # -------------------------------------
    # Add split paths to the return dict
    # -------------------------------------
    return_dict['Train number'] = len(train_list)
    return_dict['Val number'] = len(val_list)
    return_dict['Test number'] = len(test_list)
    return_dict['Sampled number'] = len(train_list) + len(val_list) + len(test_list)
    return_dict['Input number'] = len(imgPaths)
    return_dict['Sampled paths list'] = [train_list, val_list, test_list]

    # ------------------------------------
    # Copy/move files to TVT directories
    # ------------------------------------

    if not softMode:
        # Move/copy the files from the source folder (class subfolders) to the destination (class subfolder) within either the T, V or T folder.
        # Note: If the file is not found in the directory, that means it was wrongly present in the excel file (the reverse can also be true, a file may exist in the directory but not in the excel file). In this case, the file will be skipped and log an error.
        for item in train_list:
            if Path(item).is_file():
                shutil.copy2(Path(item), Path(dstPath, 'train', className, Path(item).parents[0].name+' '+Path(item).name))
            else:
                pv.core.basic_logger('File '+str(item)+' not found. Skipping.', logLevel='warning', logFileName='log.txt', logPath=dstPath, printLog=True)
        for item in val_list:
            if Path(item).is_file():
                shutil.copy2(Path(item), Path(dstPath, 'val', className, Path(item).parents[0].name+' '+Path(item).name))
            else:
                pv.core.basic_logger('File '+str(item)+' not found. Skipping.', logLevel='warning', logFileName='log.txt', logPath=dstPath, printLog=True)
        for item in test_list:
            if Path(item).is_file():
                shutil.copy2(Path(item), Path(dstPath, 'test', className, Path(item).parents[0].name+' '+Path(item).name))
            else:
                pv.core.basic_logger('File '+str(item)+' not found. Skipping.', logLevel='warning', logFileName='log.txt', logPath=dstPath, printLog=True)
        
        #    Logging
        # -------------------------------------------------------------------- #
        # Log the number of files copied along with the split ratio and the split numbers
        log_text = 'Total files provided = {}, sampled = {}, (train {}, val {}, test {}), ratio provided = {}, class name = {}'.format(len(imgPaths), len(train_list)+len(val_list)+len(test_list), len(train_list), len(val_list), len(test_list), tvtRatio, className)
        pv.core.basic_logger(log_text, logLevel='info', logFileName='info.txt', logPath=dstPath, printLog=True)
    
    return return_dict



# ============================================================================ #
#    MODELS
# ============================================================================ #


#    SEQUENTIAL DENSE MODELS
# ============================================================================ #

def dense_model_simple(
            inputShape=(1,), numberLayers=3,
            neuronsPerLayer=(16, 32, 2),
            activationPerLayer=('relu', 'relu', 'softmax'),
            modelName=''
            ):

    """
    Create a simple sequential model consisting entirely of dense layers.

    """

    # Handle inputs
    # -----------------
    # Throw an error if the size of layers etc. do not match
    if not len(neuronsPerLayer) == len(activationPerLayer) == numberLayers:
        print('Number of parameters in neuronsPerLayer and/or activationPerLayer do not match the number of hidden layers passed. Function exiting.')
        return None

    # initialize the model
    model = Sequential(name=modelName)

    # build the first layer
    model.add(Dense(units=neuronsPerLayer[0], input_shape=inputShape, activation=activationPerLayer[0], name='layer-1'))

    for i in range(numberLayers-1):

        # increment i because first layer has already been created
        i += 1

        model.add(Dense(units=neuronsPerLayer[i], activation=activationPerLayer[i], name='layer-'+str(i+1)))
    
    return model


#    SEQUENTIAL CNN MODELS
# ============================================================================ #

def cnn_model_simple(
            inputShape:tuple, numConvLayers:int=3, classes:int=2,
            filtersPerConvLayer:tuple=(16, 32, 64),
            kernelSizePerConvLayer:tuple=((3, 3), (3, 3), (3, 3)),
            stridesPerConvLayer:tuple=((1, 1), (1, 1), (1, 1)),
            poolSizePerConvLayer:tuple=((2, 2), (2, 2), (2, 2)),
            activationPerLayer:tuple=('relu', 'relu', 'relu', 'softmax'),
            modelName:str='CNN', summary:bool=True
        ):

    """
    Create a simple sequential CNN model.
    TO DO: add a rescaling layer at the top. See https://www.tensorflow.org/tutorials/images/classification.
    """

    # Handle inputs
    # -----------------
    # Throw an error if the size of layers etc. do not match
    if not len(filtersPerConvLayer) == len(kernelSizePerConvLayer) == len(stridesPerConvLayer) == len(activationPerLayer)-1 == numConvLayers:
        print('Number of filter, kernel size, strides and activation(-1) parameters do not match the number of convolution layers passed. Function exiting.')
        return None

    model = Sequential(name=modelName)

    # create first two layers that will handle the input shape and the corresponding maxpool
    model.add(Conv2D(filtersPerConvLayer[0], kernelSizePerConvLayer[0], stridesPerConvLayer[0], activation=activationPerLayer[0], input_shape=inputShape, name='conv2d-1'))
    model.add(MaxPooling2D(poolSizePerConvLayer[0], name='maxPooling-1'))

    for i in range(numConvLayers-1):
        # increment i because first layer has already been created
        i += 1
        # create the 2nd onwards convolutional layer and give it a name
        model.add(Conv2D(filtersPerConvLayer[i], kernelSizePerConvLayer[i], stridesPerConvLayer[i], activation=activationPerLayer[i], name='conv2d-'+str(i+1)))
        # create the 2nd onwards maxpool layer and give it a name
        model.add(MaxPooling2D(poolSizePerConvLayer[i], name='maxPooling-'+str(i+1)))

    # create the last Dense layer based on the number of classes
    # value of i should be what it was at the end of the loop
    model.add(Flatten(name='flatten-1'))
    model.add(Dense(classes, activation=activationPerLayer[i+1], name='dense-1'))

    if summary:
        model.summary()

    return model


def model_summary_to_string(model, delimiter:str=';'):
    
        """
        Convert the summary of a model to a pretty string.
        """
        return 'model name = ' + model.name + delimiter + delimiter.join(
            [layer.name + '(' + layer.__class__.__name__ + ') | shape = ' + str(layer.output_shape) + ' | params = ' + str(layer.count_params()) for layer in model.layers]) + delimiter + 'total params = ' +  str(model.count_params()) 



# ============================================================================ #
#    PLOTS
# ============================================================================ #


#    CONFUSION MATRIX
# ============================================================================ #

def plot_confusion_matrix(
    cm, classes='',
    title=None,
    savePath:str=None,
    displayDpi=120, saveDpi=300,
    cmap=plt.cm.Oranges, styleSheet='default',
    xTickRotation=0, yTickRotation=90
    ):
    
    """
    Creates a plot for visualizing a confusion matrix.

    ARGUMENTS:
    cm: confusion matrix to plot
    classes: list of class names. Remember that the order of the classes must match the order of the confusion matrix. Since top-left is the true-positive, the first class label should be the positive label class.

    # TODO: add figure size? Check for multiclass if needed.

    """

    # Calculate a normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Retain only two decimal places of precision
    cm_norm = np.around(cm_norm, decimals=2)

    with plt.style.context(styleSheet):

        plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
        plt.title(title)

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=xTickRotation)
        plt.yticks(tick_marks, classes, rotation=yTickRotation)

        if cm.shape[0] <= 10:
            for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
                plt.text(j, i-0.1, str(cm_norm[i, j]*100)+'%', fontsize=12, horizontalalignment='center', verticalalignment='center', color='white',
                path_effects=[pEffects.SimpleLineShadow(offset=(0,0), alpha=0.9, shadow_color='#555555', linewidth='2'), pEffects.Normal()])
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i+0.1, '('+str(cm[i, j])+')', fontsize=8, horizontalalignment='center', verticalalignment='center', color='white',
                path_effects=[pEffects.SimpleLineShadow(offset=(0,0), alpha=0.9, shadow_color='#555555', linewidth='2'), pEffects.Normal()])
            
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.grid(False)
        plt.minorticks_off()
        plt.gcf().set_dpi(displayDpi)

        if savePath:
            # save the plot to a file
            plt.savefig(Path(savePath), dpi=saveDpi)
            
        plt.show()


#    CM AND ROC THINGS
# ============================================================================ #

# based on the predictions on the test set, we can calculate CM and ROCs ab initio.

def binary_cm_roc(
    probabilities:np.ndarray,
    groundTruths:np.ndarray,
    thresholds:list
    ):

    """
    Calculates the ROC curve and the best confusion matrix for a binary classification problem using a list of custom thresholds. Returns a bunch of items including the TPR, FPR and the best confusion matrix. Class 1 is considered positive and 0 negative.
    
    ARGUMENTS:
    
    probabilities (numpy array, required): numpy array of probabilities. These are usually of the form [[0.06], [0.99], [0.01], ...] for a single neuron in the last dense layer for binary classification OR one-hot encoded probabilities for multiclass classification.
    groundTruths (numpy array, required): Known labels for the data. Should be numeric.
    thresholds (list, required): List of custom thresholds.

    RETURNS:
    A dictionary of decisions, accuracies, TPs, FPs, TNs, FNs, TPRs, FPRs, CMs, the best confusion matrix, threshold and accuracy. CM is of the form [[TP, FP], [FN, TN]], where class 1 is considered positive and 0, negative.

    """

    # initialize lists to hold important quantities per threshold
    decisions_per_t = []
    accuracy_per_t = []
    tp_per_t = [] # true positives   
    tn_per_t = [] # true negatives
    fp_per_t = [] # false positives
    fn_per_t = [] # false negatives
    cm_per_t = [] # confusion matrix
    tpr_per_t = []    # true positive rate
    fpr_per_t = []    # false positive rate
    return_dict = {}

    for t in thresholds:

        # probability (of a given sample to be the '1' class) is a number between 0 and 1 for single neuron. Also, it looks like the model.predict returns a list of lists, nx1, like [[0.06], [0.04], ...], so we need to access the probability for each row using [:, 0].
        decisions = probabilities[:, 0] > t # if probability is greater than threshold, it is a '1' class.
        # NOTE: I think for one-hot encoding the code for decisions stays the same, except that > t is replaced by < t.
        # convert decisions to ints
        decisions = decisions.astype(int)

        accuracy = np.mean(decisions == groundTruths)
        
        # this might be different for one-hot encoding
        tp = np.sum(np.logical_and(decisions == 1, groundTruths == 1)).astype(int)
        tn = np.sum(np.logical_and(decisions == 0, groundTruths == 0)).astype(int)
        fp = np.sum(np.logical_and(decisions == 1, groundTruths == 0)).astype(int)
        fn = np.sum(np.logical_and(decisions == 0, groundTruths == 1)).astype(int)

        # find the confusion matrix for this threshold
        cm = np.array([[tp, fp], [fn, tn]]).astype(int)

        tpr = tp/(tp+fn) # by definition
        fpr = fp/(fp+tn) # by definition
        
        decisions_per_t.append(decisions)
        accuracy_per_t.append(accuracy)
        tp_per_t.append(tp)
        tn_per_t.append(tn)
        fp_per_t.append(fp)
        fn_per_t.append(fn)
        cm_per_t.append(cm)
        tpr_per_t.append(tpr)
        fpr_per_t.append(fpr)

    # find the best roc point, based on its distance from the point (0,1)
    # hint: distance of (x,y) from point (0,1) = x^2 + (y-1)^2
    distances = np.square(np.array(fpr_per_t)) + np.square(np.array(tpr_per_t)-1)
    # find index of minimum distance (the first occurrence from left is returned)
    best_index = np.argmin(distances)
    # best threshold is the point of this index
    best_threshold = thresholds[best_index]

    # use the best threshold to find the best confusion matrix
    best_cm = cm_per_t[best_index]
    
    # find the best accuracy
    best_accuracy = accuracy_per_t[best_index]

    return_dict['decisions'] = decisions_per_t
    return_dict['accuracy'] = accuracy_per_t
    return_dict['tp'] = tp_per_t
    return_dict['tn'] = tn_per_t
    return_dict['fp'] = fp_per_t
    return_dict['fn'] = fn_per_t
    return_dict['cm'] = cm_per_t
    return_dict['tpr'] = tpr_per_t
    return_dict['fpr'] = fpr_per_t
    return_dict['distances'] = distances
    return_dict['best index'] = best_index
    return_dict['best threshold'] = best_threshold
    return_dict['best cm'] = best_cm
    return_dict['best accuracy'] = best_accuracy

    return return_dict