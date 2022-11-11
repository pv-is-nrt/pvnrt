#
# ============================================================================ #
#
#
#   Module of some core functions
#   Author: Prateek Verma
#   Created on: July 12, 2021
#
#
# ============================================================================ #
#


#
# ============================================================================ #
#
#    COMMON IMPORTS
#
# ============================================================================ #
#

import os, shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tqdm import tqdm
import numpy as np
import math
import ctypes, sys


#
# ============================================================================ #
#
#               FILESYSTEM RELATED FUNCTIONS
#
# ============================================================================ #
#

#    WINDOWS ADMIN CHECK
# -------------------------------------------------------------------- #

def is_admin():

    """
    This is how to use it
    if is_admin():
        # your program
        print('You are an admin')
    else:
        # Re-run the program with admin rights
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False



#    SEARCH FOR FILES
# -------------------------------------------------------------------- #
def filetype_search(src_path, file_exts, verbose=0) -> list:

    """
    Search for files with given extension(s) in a given directory and returns a list of file paths.

    DEPENDS ON:
    OS & datetime modules.
    
    ARGUMENTS:

      src_path (string, required): is the path to a directory. Make sure to construct the path using appropriate string or path building functions prior to passing in to this function. It can relative or absolute and will decide whether the output list of paths will contain relative or absolute paths.

      file_exts (string, tuple or list, required): are the file extensions (lowercase and without the .) that are desired to be searched for. If there is only one extension, a string can be passed instead of a tuple/list. Filenames ending with the exact string are returned.

      verbose (booloean, optional): prints a list of files if set to True, and prints only function status messages if set to False (default).

    RETURNS:
    A list containing the paths to the found files. Paths are absolute or relative depending on what was passed to the function.

    For example,
    the following usage returns prescribed image files in the windows path provided and saves it in a list called src_file_list.

        SRC_PATH = '..\Demo Images\imageData2700'
        FILE_EXTS = ['png', 'bmp', 'tif', 'tiff', 'jpg', 'jpeg']
        src_file_list = pv.core.filetype_search(SRC_PATH, FILE_EXTS, verbose=2)

    """
    
    start_time = datetime.now().timestamp() # timer to measure how long the function ran for
    
    if verbose>0: 
        print('Searching for files with extension(s) {} in\n    {}'.format(file_exts, src_path))

    # counters for files and subfolders
    TOTAL_FILES = 0
    TOTAL_SUBFOLDERS = 0

    # convert passed path to a Path object, it's not necessary, the code will work even without this step, but converting it to a Path object gives us access to Path methods.
    src_path = Path(src_path)

    # initialize a list of eligible paths, which will be the output of the function
    paths = []

    for root, subfolders, files in os.walk(src_path):
        # walks inside the root looking at each subfolder and file
        
        for subfolder in subfolders:
            TOTAL_SUBFOLDERS += 1 # just count the number of subfolders
        
        for file in files:
            # split the file at the extension . and return the last element of the list (it's a str), convert it to lowercase and see if the str extension is part of the passed extension list/touple/string or not
            if file.split('.')[-1].lower() in file_exts:
                TOTAL_FILES += 1
                # path will be os independent and will start at the beginning of the provided src_path
                # root here will include all the subdirectories that a file is in (IMPORTANT COMMENT)
                # thus it is not needed to append subfolder to the path
                # if you append (use) subfolder, and there are no subfolders in the src_path, it will raise an error (IMPORTANT COMMENT) 
                paths.append(os.path.join(root, file))
                if verbose>1:
                    print('   {}'.format(os.path.join(root, file)))

    if verbose>0:
        print('Walked through {} subfolders and found {} matching files [Processing time {} ms].'.format(TOTAL_SUBFOLDERS, TOTAL_FILES, processing_time(start_time)))
    
    return paths

#    REMOVE HEADS FROM FILE PATHS
# -------------------------------------------------------------------- #
def remove_path_head(filePaths:list, levels:int):
    """
    Removes the specified number of levels from the head of the filepaths.
    RETURNS: A list of Windows Path objects.
    """
    new_file_paths = []
    for fyle in filePaths:
        # convert to pathlib object
        fyle = Path(fyle)
        # clean up .. in the file path if any
        new_path = [item for item in fyle.parts if item != '..']
        # drop the first few levels
        new_path = new_path[levels:-1]
        # convert the list to a path
        new_path = Path(*new_path)
        # add filename to the path
        new_path = Path(new_path, fyle.name)
        new_file_paths.append(new_path)
    return new_file_paths

#    GET GLOB PATH LIST
# -------------------------------------------------------------------- #
def get_glob_paths_list(path:str, glob:str):

    """
    Get list of paths from a given path (common path for all the intended globs) with a given glob string. See pathlib.Path.glob() for more details.

    Returns a list if multiple paths are found, or a single path if only one is found.
    """

    glob_list = list(Path(path).glob(glob))
    return glob_list if len(glob_list)>1 else glob_list[0]

#    FIND NEXT ID
# -------------------------------------------------------------------- #
def get_next_id(rootDir:str, removePrefix:str, lenID:int, fileOrDir:str='dir'):

    """
    Finds the next ID (number, largest) from the given directory. The directory is expected to have folders with names containing an ID number of a fixed length. The ID number must be an integer. Maximum ID number is found and 1 is added to it and returned.

    ARGUMENTS:
      
      rootDir (string, required): Path to the folder inside which you wish to scan for IDs.

      removePrefix (string, required): Part of the folder name that you wish to remove from the left to where the ID starts. Be sure to include spaces, hyphens etc. that need to be removed. They are not removed automatically.

      lenID (integer, required): number of digits that comprise the IDs. All IDs must be of the same length. The returned ID will also contain the same number of digits with leading zeroes.

    RETURNS: A new string (not integer) ID (unique, incremented the maximum ID found in the folder by 1) is returned with the same length as the lenID, with enough leading zeroes.

    """

    # Check whether it is specified whether to check for files or directories for IDs
    # Make sure it is one of them
    if fileOrDir!='dir' and fileOrDir!='file':
        print("ERROR: The value of the parameter fileOrDir must be one of 'file' or 'dir'. Function exiting.")
    
    path = Path(rootDir)
    IDs = [0]
    for item in path.glob('*'):
        if fileOrDir=='dir':
            if item.is_dir():
                ID = item.name.removeprefix(removePrefix) if item.name[:len(removePrefix)]==removePrefix else None
                if ID:
                    ID = int(ID[:lenID])
                    IDs.append(ID)
        else:
            if item.is_file():
                ID = item.name.removeprefix(removePrefix) if item.name[:len(removePrefix)]==removePrefix else None
                if ID:
                    ID = int(ID[:lenID])
                    IDs.append(ID)
    newID = max(IDs) + 1
    newID = str(newID).zfill(lenID)

    return newID

#    FRIENDLY SIZE FROM BYTES
# -------------------------------------------------------------------- #
def friendly_size(sizeBytes):
    """
    This function will return a friendly size from bytes
    """
    if sizeBytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(sizeBytes, 1024)))
    p = math.pow(1024, i)
    s = round(sizeBytes / p, 2)
    return "%s %s" % (s, size_name[i])

#    TOTAL FILE SIZE AND COUNT OF A DIRECTORY
# -------------------------------------------------------------------- #
def get_dir_size_file_count(path):
    """Return total size of files and total number of files in given path and subdirs."""
    size = 0
    file_count = 0
    try:
        for entry in os.scandir(path):
            if entry.is_dir(follow_symlinks=False):
                s, c = get_dir_size_file_count(entry.path)
                size += s
                file_count += c
            else:
                size += entry.stat(follow_symlinks=False).st_size
                file_count += 1
    except:
        print('ERROR: problems reading directory {}'.format(path))
    return size, file_count

#    PRINT DIR TREE SIZE TABLE
# ---------------------------------------------------------------------------- #
def get_tree_size_df(srcFolder, progressBar=(80,'▢▣','#CC6655'), removeCommonPath=True):
    """
    This function returns a pandas dataframe with all the subdirectories and the total size of each subdirectory in a given source folder.

    ARGUMENTS:

        srcFolder (string, required): Path to the source folder.

        cleanDF (boolean, optional): If True, the returned dataframe will be cleaned of path parts from root to source.

    RETURNS: A pandas dataframe with a column for each folder in path, a size column and a column showing total files within each folder.

    TODO:
    - Use single recursion instead of two. Glob in addition to the recursive scandir in inefficient, even though it may be fast enough for usage. For example, on a subdir ~batches in HDD3, it took 88 mins.
    - Ignore dirs starting with '.' or system directories. Or is there a way to ignore dirs that the script is not allowed to access?

    """
    master_table = []
    size_column = [] # because the row length will be variable, we will need to add the size column after df is created so it can be in its own column.
    # similarly
    size = 0
    num_files = 0
    size_gb_column = []
    num_files_column = []
    src = Path(srcFolder)
    list_root_paths = list(src.glob('**/'))
    for path in tqdm(list_root_paths, ncols=progressBar[0], ascii=progressBar[1], colour=progressBar[2]):
        size, num_files = get_dir_size_file_count(path)
        # get size in MBs and GBs so they can be sorted in excel file
        size = round(size / (1024 * 1024), 2)
        size_gb = round(size / 1024, 2)
        size_column.append(size)
        size_gb_column.append(size_gb)
        num_files_column.append(num_files)
        master_row = [path.name]
        for parent in path.parents:
            master_row.append(parent.name)
        # reverse parent_list
        master_row.reverse()
        master_table.append(master_row)
    # convert master_table to a dataframe
    df = pd.DataFrame(master_table)
    df['size (MB)'] = size_column
    df['size (GB)'] = size_gb_column
    df['total files'] = num_files_column
    # replace None with empty string so the unique function works
    df.fillna('', inplace=True)
    if removeCommonPath:
        nunique = df.nunique()
        cols_to_drop = nunique[nunique == 1].index
        df.drop(cols_to_drop, axis=1, inplace=True)
    return df


#
# ============================================================================ #
#
#               PYTHON FUNCTIONS
#
# ============================================================================ #
#

#    GET KEYS FROM VALUE IN A DICT 
# -------------------------------------------------------------------- #
def get_keys_from_value(dictionary:dict, value)->list:

    """
    
    Simple function to return a list of keys in the provided dictionary that correspond to the provided value.

    """

    return [key for key, val in dictionary.items() if value == val]

#    BASIC LOGGING
# -------------------------------------------------------------------- #
def basic_logger(
    logString:str,
    logLevel:str='info',
    logFileName:str='log.txt', 
    logPath:str='',
    printLog:bool=True,
    ) -> None:

    """
    A simple function to append a log string to a file.

    ARGUMENTS:

      logString (string, required): The string to be logged.

      logLevel (string, optional): The level of the log. Can be 'info', 'warning', 'error' or 'critical'. Default is 'info'.

      logPath (string, optional): The path to the folder where the log file will be saved. Defaults to the current working directory.

      logFileName (string, optional): The name of the log file. Defaults to 'log.txt'.

    RETURNS: Formatted log string.

    """

    # convert the file path to a Path object
    log_file_path = Path(logPath, logFileName)

    # format the logString as needed
    # add a timestamp and a level
    logString = '[' + logLevel.upper() + '] ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': ' + logString + '\n'

    # Make sure the path exists
    if not log_file_path.parent.is_dir():
        os.makedirs(log_file_path.parent)

    with open(log_file_path, 'a') as fyle:
        fyle.write(logString)

    if printLog:
        print(logString)


#
# ============================================================================ #
#
#               TIME FUNCTIONS
#
# ============================================================================ #
#

#    EVALUATE PROCESSING TIME
# -------------------------------------------------------------------- #

def start_time():
    """
    Returns the current time in seconds
    """
    return datetime.now().timestamp()

def processing_time(start_time:float, stop_time='now', units:str='ms', decimals:int=3) -> float:
    
    """
    Evaluates and returns the duration of time elapsed between two time stamps or between a given time stamp and when the function is called.

    DEPENDS ON:
    Datetime module.

    ARGUMENTS:

      start_time (required): timestamp at the beginning.
        
      stop_time (optional): timestamp at the end. Default is whenever the function is called. You will not have to specify the keyword 'now' to use the default.

      units (string, optional, default is ms (milliseconds)): is a string defining the units of the returned time. Allowed values are mins, min or m (for minutes), s (for seconds), ms (for milliseconds) or um (for microseconds).

      decimals (integer, optional, default is 3): number of digits the returned time will be rounded to.

    RETURNS:
    The processing time in the desired units and rounded to the desired decimal places.

    For example, the following usage eturns the number of milliseconds passed between the time when 'time' was defined and the function was called, accurate to 3 decimal places.

        time = datetime().now().timestamp()
        processing_time(time)

    """

    # if the stop time is passed as the keyword now then convert that to the current timestamp. For some reason passing it inside the function definition argument did not work.
    if stop_time == 'now':
        stop_time = datetime.now().timestamp()
    
    # Return the time according to the units, decimals
    if units == 'ms':
        return round((stop_time-start_time)*1000, decimals)
    elif units == 's':
        return round((stop_time-start_time), decimals)
    elif units == 'us':
        return round((stop_time-start_time)*1000000, decimals)
    elif units == 'min' or units == 'm' or units == 'mins':
        return round((stop_time-start_time)/60, decimals)
    else:
        print('Time units not defined correctly. Allowed values are mins, min or m (for minutes), s (for seconds), ms (for milliseconds) or um (for microseconds)')
        return None


#
# ============================================================================ #
#
#               IMAGE RELATED FUNCTIONS
#
# ============================================================================ #
#

#    IMAGE RESIZER
# -------------------------------------------------------------------- #
def image_resize(
    filePaths:list, 
    width:int, height:int,
    mode='contain',
    dstPath='resized',
    discardSrcLevels:int=0,
    dstFileFormat=None,
    dstFileExtension=None,
    returnOnly=True,
    clearDstDir=False,
    overWrite=True,
    progressBar=(80,'▢▣','#CC6655'),
    verbose=0
    ):

    """
    
    ARGUMENTS:
    
      filePaths (list, required): A list of paths of files meant to be resized. Files will not be modified in any way.

      width (int, one of width or height is required) Target width of the image in pixels. If set to auto, utilizes the provided height parameter and sets a width that maintains the aspect ratio of the image.

      height (int, one of width or height is required) Target height of the image in pixels. If set to auto, utilizes the provided width parameter and sets a height that maintains the aspect ratio of the image.

      mode (string, optional): Choose among 'fit' or 'contain'. Please see the documentation for Pillow's ImageOps module for details on these methods. In short, fit will resize and crop the image from the center in order to fit the image within the given width AND height while maintaining the aspect ratio. And, contain will resize the image within the given width or height, depending on which dimension keeps to maintain the aspect ratio intact. Default is 'contain'.
      
      dstPath (string, optional): Let's you choose the destination folder for the resized images. If 'source', then files are saved in their respective original source folder(s). If any thing other than 'source' is provided, then all the resized files will be saved to the specified folder (path). In this scenario, the directory structure of the input files is maintained within the specified destination folder. If the user is aware that the input file paths have a common root upto a certain level, then the user can specify the parameter 'discardSrcLevels' to discard a certain number of levels while replicating the dir structure of the input file paths. Default dstPath is 'resized', which means that a folder of this name will be created in the directory this script is running from and the files be stored within.

      discardSrcLevels (int, optional): Applicable only when a dstPath other than 'source' is specified. If the user is aware that the input file paths have a common root upto a certain level, then the user can specify a velue to this parameter to discard a certain number of levels while replicating the dir structure of the input file paths. For example, if the input file path is '../a/b/c/d/e/image.png' and the user knows that they only want to replicate the source dirs from 'd' onwards, since the paths up to '../a/b/c/' are common, then the user should specify discardSrcLevels = 3. The default value is 0, which means the entire source path (excluding the ..s) will be replicated. 

      dstFileFormat (string, optional): Specifies the format of the saved file as per the PIL library. If no format is provided, PIL tries to detect the format from the filename extension. Default is None, which means that the format will be detected from the input image.

      dstFileExtension (string, optional): Specifies the extension of the saved file. If no extension is provided, files extension will be used. Default is None.

      returnOnly (boolean, optional): Gives the user the option to also save the files to disk (False) in addition to returning the images as a PIL image object. Default is True, which means that images will only be returned by the function and not actually saved to the disk.

      clearDstDir (boolean, optional): Applicable only to the case where dstPath != 'source'. If True, it deletes the specified destination folder and everything within it before re-creating it and copying files to it. Note that this does not work when the destination directories are the source folders themselves (consider it a pending feature). Default is True.

      overWrite (boolean, optional): Applicable when the returnOnly parameter is set to False. If True, files with the same name will be overwritten. If False, files with the same name will be skipped.

      progressBar (tuple, optional): Specifies the formatting and size of the tqdm progress bar. Specify the tuple as (length of bar expressed as columns : int, ascii characters to form the bar : character sequence as string, color : color). Default is (80,'▢▣','#CC6655').

      verbose (boolean, optional): Allows user to specify the verbosity of logging prints on screen. Default is 0 (minimum verbosity).

    TO DO:
    - currently, unable to clear 'resized' directory if user chooses destination as 'source' folder. This is because the logic is a tad bit complicated as noted in the comments.
    - Don't read the files that already exist in the resized folder

    """


    # ==================================================================== #
    #    Initialize things
    # ==================================================================== #
    
    return_images = {} # key can correspond to the original image path
    skipped = 0


    # ==================================================================== #
    #    PREPARE PATH(S)
    # ==================================================================== #

    # Make sure that the source file paths are coming in as list and are formatted as Path objects
    if type(filePaths) != list:
        print('ERROR: Argument filePaths must be passed in as a list. Function exiting.')
        return None
    file_paths = [Path(item) for item in filePaths]

    # If the destination is same as source folder (specified by the value 'source')
    if dstPath == 'source':
        # PLACEHOLDER: this condition will be tackled for each image inside the loop
        None
    else:
        # PLACEHOLDER: some of this condition will be tackled for each image inside the loop
        # but we can at least convert the path to a Path object here
        dst_path = Path(dstPath)
        if dst_path.is_dir()==False:
            if returnOnly==False:
                os.makedirs(dst_path)
        else:
            if clearDstDir:
                if returnOnly==False:
                    # remove directory and everything inside if user so desires
                    shutil.rmtree(dst_path)
                    # then create an empty directory
                    os.mkdir(dst_path)
            # if clearDstDir is passed as false, then the dir with its old contents will keep existing and the files in there will be overwritten or added to, and the user will probably know what is going on when looking into the desired destination dir


    # ==================================================================== #
    #    RESIZE >> SAVE OR RETURN
    # ==================================================================== #
    
    # loop through each image path
    for fyle in tqdm(file_paths, ncols=progressBar[0], ascii=progressBar[1], colour=progressBar[2]):

        # open image using the PIL Image library
        with Image.open(fyle) as image:
            
            #    FIT MODES
            # -------------------------------------------------------------------- #
            if mode=='fit':
                new_image = ImageOps.fit(image, (width, height))
            if mode=='contain':
                new_image = ImageOps.contain(image, (width, height))
            
            #    HANDLE SAVE/RETURN FOR EACH IMAGE
            # -------------------------------------------------------------------- #
            # If user has chosen to save the files to disk
            if not returnOnly:
                # If user has chosen to save file in the same folder as the original
                # Note that we don't need to check if parent directories exist
                if dstPath=='source':
                    # check whether the 'resized' directory exists or not, if not create it
                    # this condition will only be false for the first file iteration
                    if Path(fyle.parent, 'resized').is_dir()==False:
                        os.mkdir(Path(fyle.parent, 'resized'))
                    # IMPORTANT - clearing the destination directory in this case is a complicated process and hence skipping it for now. That is, if files are chosen to be placed inside their source folders, the 'resized' folder within will NOT be cleared before placing files there. The process is complicated because we want the folder to not be cleared on every file loop. We cannot keep this check outside the loop either, because each file's parent is probably different. 
                    
                    # Now check whether the file exists or not and whether overwrite is set to true or not
                    if Path(fyle.parent, 'resized', fyle.name).is_file():
                        if overWrite:
                            new_image.save(Path(fyle.parent, 'resized', fyle.name if dstFileExtension == None else fyle.stem+'.'+dstFileExtension), dstFileFormat)
                        else:
                            skipped +=1
                            if verbose>1:
                                print('File', fyle.name, 'already exists. Overwrite not permitted. Skipping.')
                    else:
                        new_image.save(Path(fyle.parent, 'resized', fyle.name if dstFileExtension == None else fyle.stem+'.'+dstFileExtension), dstFileFormat)
                # If user has provided a destination folder path
                else:
                    # Remember that the file paths coming in can be of all kinds of lengths and may have no dir in common (unlikely though). Since we do not have a common source path, it it not possible to easily create a simple source structure by finding the common dir where all source file paths emerge from. For example, if all our files reside in D in the path '..\A\B\C\D\E,F,G\images, it will not be easy to figure out D just from the file paths. (1) We will have to recreate the entire source structure starting from A. User will have to manually delete A\B\C after the fact from file explorer, which is deemed doable. (2) Another alternative is to let the user provide a starting depth after which to start the source structure creation.
                    # Let's implement the above idea.
                    desired_src_structure = []
                    # This will extract parts of the source file paths devoid of '..'s
                    desired_src_structure = [item for item in fyle.parts if item != '..']
                    # this will keep only the path parts that start from the desired depth. Default is 0, which means that all parts will be kept. -1 is used to discard the filename itself.
                    desired_src_structure = desired_src_structure[discardSrcLevels:-1]
                    # convert the list to a path object and overwrite the list
                    desired_src_structure = Path(*desired_src_structure)
                    if Path(dst_path, desired_src_structure).is_dir()==False:
                        # create the directories where the resized image will reside
                        os.makedirs(Path(dst_path, desired_src_structure))
                        new_image.save(Path(dst_path, desired_src_structure, fyle.name if dstFileExtension == None else fyle.stem+'.'+dstFileExtension), dstFileFormat)
                    else:
                        # since the destination dir exists, check whether the file exists or not and if overwrite is set to true or not
                        if Path(dst_path, desired_src_structure, fyle.name).is_file():
                            if overWrite:
                                new_image.save(Path(dst_path, desired_src_structure, fyle.name if dstFileExtension == None else fyle.stem+'.'+dstFileExtension), dstFileFormat)
                            else:
                                skipped +=1
                                if verbose>1:
                                    print('File', fyle.name, 'already exists. Overwrite not permitted. Skipping.')
                        else:
                            new_image.save(Path(dst_path, desired_src_structure, fyle.name if dstFileExtension == None else fyle.stem+'.'+dstFileExtension), dstFileFormat)
            
            # return the dict of files whether or not the user has chosen to save the files
            return_images[fyle] = new_image

    # If saving and overwriting is False
    if overWrite==False and returnOnly==False and clearDstDir==False:
        if verbose>0:
            print('Skipped', skipped, 'number of files, because they already existed on disk.')

    return return_images


#
# ============================================================================ #
#
#               DATAFRAMES RELATED FUNCTIONS
#
# ============================================================================ #
#

#    DATAFRAMES TO EXCEL
# -------------------------------------------------------------------- #

def dataframes_to_new_excel(dataframes:list, sheetnames:list, fullFileName:str, overwrite:bool=False, verbose=0) -> None:

    """
    Creates an excel file from a dataframe. If multiple dataframes are passed, it creates as many sheets inside the excel file.

    DEPENDS ON:
    Pathlib, Excelwriter, xlsxwriter, datetime etc. 

    ARGUMENTS:

      dataframes (type dataframe, required): either a single dataframe or a list of dataframe objects. If it's a list, make sure that equal number of sheet names are passed.

      sheetnames (string or list, required): either a single string or a list (containing elements equal to the number of dataframes) of strings.

      fullFileName (string, required): should be the complete file name of the excel file to be created, including the file extension and the directory path (relative or absolute).

      overwrite (bool, optional): if set to True, the function will overwrite the file if it already exists. Else it will ask the user to confirm overwriting the file. Default is False.
    
      verbose (int, optional): Prints out details of the sheets created. Default is 0.

    RETURNS:
    Nothing.
        
    """

    # make sure that the length of dataframes and sheetnames match
    # if they are a list and not a string
    if type(dataframes) == list and type(sheetnames) == list:
        if len(dataframes) != len(sheetnames):
            print('Number of elements in the dataframes ({}) does not match the number of elements in the sheetnames ({}). Function exiting.'.format(len(dataframes), len(sheetnames)))
            return None
    elif type(dataframes) == pd.DataFrame and type(sheetnames) == str:
            # enclose dataframe object and string inside a list
            dataframes = [dataframes]
            sheetnames = [sheetnames]
    else:
        print('ERROR: Both dataframes and sheetnames should be of type list or dataFrame and str. Function encountered a type mismatch. Function exiting.')
        return None

    # convert the passed fullFileName into a path
    path = Path(fullFileName)

    if path.is_file() and overwrite==False:
        # ask user if they wish to overwrite or create a new file, in case the file already exists.
        userInput = input('ALERT: target Excel file already exists. Do you want to proceed and overwrite the data (y)? Or create a new file (n)?')
        if userInput != 'y':
            # Create a new filename by adding current timestamp as a suffix
            stamp = round(datetime.now().timestamp())
            path = Path(path.parent, '{} dup{}{}'.format(path.stem, stamp, path.suffix))

    start_time = datetime.now().timestamp() # timer to measure how long the writer function ran for

    # check if the output folder (contained in the fullFileName) exists
    if path.parent.exists():
        # create file
        # first create an excel writer object in pandas
        # note that xlsxwriter will need to be installed via pip
        excelWriter = pd.ExcelWriter(path)
        # then write the passed dataframe(s) to the corresponding excel sheet(s)
        for i in range(len(dataframes)):
            dataframes[i].to_excel(excelWriter, sheet_name=sheetnames[i])
            if verbose:
                print('Written to sheet:{}'.format(sheetnames[i]))
        excelWriter.save()
        print('File {} created successfully'.format(path))
    else:
        print('ERROR: Directory {} does not exist. Please create the directory and try again'.format(path.parent))
        return None

    print('Time taken to create excel file(s) = {} ms'.format(processing_time(start_time)))


#
# ============================================================================ #
#
#               PLOTTING FUNCTIONS
#
# ============================================================================ #
#

#    SIMPLE PLOT
# -------------------------------------------------------------------- #
def plot_simple(
    xDataList, yDataList, seriesLabels=None,
    markers=None, colors=None,
    xLabel=None, yLabel=None, title=None,
    xLimits=None, yLimits=None,
    legendLocation='best',
    styleSheet='default',
    figSize=(3.2, 2.4), 
    savePath=None,
    displayDPI=150, saveDPI=300
    ) -> plt:

    """
    A simple function to plot a scatter style plot of Y vs X, offering quick access to customizing labels, title, axis limits and style sheet.

    ARGUMENTS:

      xDataList (list(s), required): A list or lists containing x-axis data. If a single xData list is passed, make sure it is passed inside a list.

      yDataList (list(s), required): A list or lists containing y-axis data. If a single yData list is passed, make sure it is passed inside a list. For example, [history['accuracy']] and not just history['accuracy'].

      seriesLabels (list of strings, optional): Label(s) for y that gets printed on the Legend. Try and provide label for each yData. In case of length mismatch between yData and seriesLabels, the shorter list will determine the number of plots. Nothing is printed by default.

      markers (list of strings, optional): Marker(s) for each yData. If not provided, the default marker is 'o'.

      colors (list of strings, optional): Color(s) for each yData. If not provided, the default colors from stylesheet will be used.

      xLabel (string, optional): Label for x-axis that gets printed next to the axis. Nothing is printed by default.

      yLabel (string, optional): Label for y-axis that gets printed next to the axis. Nothing is printed by default.

      title (string, optional): Title for the plot. Nothing is printed by default.

      xLimits (list of two numbers, optional): Numbers that set the limits for the minimum and maximum values shown on the x-axis. Auto axis scaling by default.

      yLimits (list of two numbers, optional): Numbers that set the limits for the minimum and maximum values shown on the y-axis. Auto axis scaling by default.

      styleSheet (string path, optional): Path to matplotlib style sheet to be applied to the current plot. Multiple stylesheets can be passed, as a list of paths (overlapping styling in the latter overrides the former). If not specified, the 'default' styling is used.

      figSize (Tuple of two numbers, optional): Size of the figure in inches. Default is (3.2, 2.4).

      savePath (string path, optional): Path to save the figure. By default, the figure is not saved to disk.

      displayDPI (int, optional): DPI of the figure to be displayed. Default is 150.

      saveDPI (int, optional): DPI of the saved figure. Default is 300.

    RETURNS: The pyplot module. This enables the plot to be modified by the user even after calling the function.

    """

    plt.figure(figsize=figSize)
    
    # create a list of xData to match the length of yData list
    if len(xDataList)==1:
        xDataList = [xDataList[0] for i in range(len(yDataList))]
    # create a list of blank series labels to match the number of yData
    if not seriesLabels:
        seriesLabels = ['']*len(yDataList)
    # create a list of markers to match the number of yData
    if not markers:
        markers = ['o']*len(yDataList)

    with plt.style.context(styleSheet):
        if colors:
            for x, y, l, m, c in zip(xDataList, yDataList, seriesLabels, markers, colors):
                plt.plot(x, y, marker=m, color=c, label=l)
        else:
            for x, y, l, m in zip(xDataList, yDataList, seriesLabels, markers):
                plt.plot(xDataList, y, marker=m, label=l)
    
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.legend(loc=legendLocation)
    plt.gcf().set_dpi(displayDPI)

    # sets the scaling for x and y axes if passed
    if type(xLimits) == list and len(xLimits) == 2:
        plt.xlim(xLimits[0], xLimits[1])
    if type(yLimits) == list and len(yLimits) == 2:    
        plt.ylim(yLimits[0], yLimits[1])

    # saves the figure if savePath is specified
    if savePath:
        plt.savefig(savePath, dpi=saveDPI, bbox_inches='tight')

    plt.show()

    return plt


#    PLOT IMAGES WITH LABELS
# -------------------------------------------------------------------- #
def plot_images_labels(
    images:list,
    cmap:str='viridis',
    labels:list=None,
    numColumns:int=6, 
    figWidth=6, figExtraHeight=0,
    textPos:tuple = (0.5, 0.2), fontSize = 'large', fontColor = 'black',
    savePath:str = None,
    styleSheet='default'
    ):

    '''
    Plots images and labels in a grid using matplotlib library.
    ARGUMENTS:
        images (list, required): list of images to be plotted. These images can be in any format that matplotlib can handle.
        lables (list, required): list of labels as strings. It is okay to provide no labels (None (default) or [] both yield blank string labels).
        numColumns (int, optional): number of columns in the grid. Default is 6.
        figWidth (int, optional): width of the figure in inches. Default is 6.
        figExtraHeight (int, optional): extra height of the figure in inches. Default is 0.
        textPos (tuple, optional): position of the text in the figure in the format 'x-position from left, y-position from bottom' and assuming that the width/height of an image scales to 1. Default is (0.5, 0.2).
        fontSize (str, optional): font size of the text. Default is 'large'.
        fontColor (str, optional): font color of the text. Default is 'black'.
        savePath (str, optional): path to save the figure. Default is None, meaning that the image will not be saved, just shown.
        styleSheet (str, optional): name of the style sheet to be used. Default is 'default'.
    RETURNS:
        None
    '''
    
    # find the number of images
    num_images = len(images)
    # make sure that the function can handle the labels if none are provided
    if labels == [] or labels == None:
        labels = [''] * num_images
    
    # Calculate number of rows based on the number of columns provided
    num_rows = int(np.ceil(num_images/numColumns))

    # generate the plot
    with plt.style.context(styleSheet):

        figure, axes = plt.subplots(num_rows, numColumns, figsize=(figWidth, figWidth*num_rows/numColumns+figExtraHeight))
        
        # flatten the 2D axes array to a 1D array, so that axes can be hidden and called in a loop and zipped
        axes = axes.flatten()
        for axis in axes:
            # this will turn each axis off, all of them, (IMPORTANT) not just the ones that contain images.
            axis.axis('off')
        
        for image, label, axis in zip(images, labels, axes):
            # show the image
            axis.imshow(image, cmap=cmap)
            # add label
            axis.text(textPos[0], textPos[1], label, fontsize=fontSize, color=fontColor, horizontalalignment='center', verticalalignment='center', transform=axis.transAxes)
        
        # save the figure if a path is provided
        if savePath != None:
            plt.savefig(savePath)
        else:
            plt.tight_layout()
            plt.show()