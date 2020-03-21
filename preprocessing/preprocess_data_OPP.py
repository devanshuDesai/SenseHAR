import os
import zipfile
import argparse
import numpy as np
import pickle as cp
from collections import Counter
from io import BytesIO
from pandas import Series
from scipy import stats

# Hardcoded number of sensor channels (5*9)
NB_SENSOR_CHANNELS = 45

# Hardcoded names of the files defining the OPPORTUNITY data. As named in the original data.
OPPORTUNITY_TRAIN_DATA_FILES = [
                          'OpportunityUCIDataset/dataset/S1-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S1-ADL5.dat',
                          'OpportunityUCIDataset/dataset/S1-Drill.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S2-Drill.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S3-Drill.dat',
                          'OpportunityUCIDataset/dataset/S4-ADL1.dat',
                          'OpportunityUCIDataset/dataset/S4-ADL2.dat',
                          'OpportunityUCIDataset/dataset/S4-ADL3.dat',
                          'OpportunityUCIDataset/dataset/S4-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S4-ADL5.dat',
                          'OpportunityUCIDataset/dataset/S4-Drill.dat',
                          ]

OPPORTUNITY_TEST_DATA_FILES = ['OpportunityUCIDataset/dataset/S2-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S2-ADL5.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL4.dat',
                          'OpportunityUCIDataset/dataset/S3-ADL5.dat'
                          ]

def select_columns_opp(data):
    """Selection of the 45 columns employed in the OPPORTUNITY challenge

    :param data: numpy integer matrix
        Sensor data (all features)
    :return: numpy integer matrix
        Selection of features
    """

    #                     included-excluded
    data = data[:, np.r_[37:46, 50:59, 63:72, 76:85, 89:98, 243, 249]]
    print(data.shape)

    return data


def divide_x_y(data, label):
    """Segments each sample into features and label

    :param data: numpy integer matrix
        Sensor data
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numpy integer array
        Features encapsulated into a matrix and labels as an array
    """

    data_x = data[:, :45]
    if label not in ['locomotion', 'gestures', 'both']:
            raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, 45]  # Locomotion label
        data_y = np.expand_dims(data_y, axis=1)
    elif label == 'gestures':
        data_y = data[:, 46]  # Gestures label
        data_y = np.expand_dims(data_y, axis=1)
    elif label == 'both':
        data_y = data[:, 45:47]  # Gestures label

    return data_x, data_y


def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]

    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17

    elif label == 'both':
        data_y[:,0][data_y[:,0] == 4] = 3
        data_y[:,0][data_y[:,0]== 5] = 4
        data_y[:,1][data_y[:,1] == 406516] = 1
        data_y[:,1][data_y[:,1] == 406517] = 2
        data_y[:,1][data_y[:,1] == 404516] = 3
        data_y[:,1][data_y[:,1] == 404517] = 4
        data_y[:,1][data_y[:,1] == 406520] = 5
        data_y[:,1][data_y[:,1] == 404520] = 6
        data_y[:,1][data_y[:,1] == 406505] = 7
        data_y[:,1][data_y[:,1] == 404505] = 8
        data_y[:,1][data_y[:,1] == 406519] = 9
        data_y[:,1][data_y[:,1] == 404519] = 10
        data_y[:,1][data_y[:,1] == 406511] = 11
        data_y[:,1][data_y[:,1] == 404511] = 12
        data_y[:,1][data_y[:,1] == 406508] = 13
        data_y[:,1][data_y[:,1] == 404508] = 14
        data_y[:,1][data_y[:,1] == 408512] = 15
        data_y[:,1][data_y[:,1] == 407521] = 16
        data_y[:,1][data_y[:,1] == 405506] = 17

    return data_y


def check_data(data_set):
    """Try to access to the file and checks if dataset is in the data directory
       In case the file is not found try to download it from original location

    :param data_set:
            Path with original OPPORTUNITY zip file
    :return:
    """
    print ('Checking dataset {0}'.format(data_set))
    data_dir, data_file = os.path.split(data_set)
    # When a directory is not provided, check if dataset is in the data directory
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
        if os.path.isfile(new_path) or data_file == 'OpportunityUCIDataset.zip':
            data_set = new_path

    # When dataset not found, try to download it from UCI repository
    if (not os.path.isfile(data_set)) and data_file == 'OpportunityUCIDataset.zip':
        print ('... dataset path {0} not found'.format(data_set))
        import urllib.request
        origin = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
        )
        if not os.path.exists(data_dir):
            print ('... creating directory {0}'.format(data_dir))
            os.makedirs(data_dir)
        print ('... downloading data from {0}'.format(origin))
        urllib.request.urlretrieve(origin, data_set)

    return data_dir

def sliding_window(dataset,labels, window, stride):
    
    n_samples, d = dataset.shape
    data_slide = np.zeros((int((n_samples-window)/stride)+1,window,d))
    labels_slide = np.zeros((int((n_samples-window)/stride)+1,1))
    k=0
    
    for i in range(0,n_samples-window,stride): 
        data_slide[k,:,:] = dataset[i:i+window,:]
        m = stats.mode(labels[i:i+window])
        labels_slide[k] = m[0]
        k=k+1

    print (data_slide.shape)
    print(labels_slide.shape)
    return data_slide, labels_slide


def process_dataset_file(data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ["gestures", "locomotion", "both"(default)]
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    data = select_columns_opp(data)

    # Colums are segmentd into features and labels
    data_x, data_y =  divide_x_y(data, label)
    print(data_x.shape)
    print(data_y.shape)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    return data_x, data_y


def generate_data(dataset, output, label):
    """Function to read the OPPORTUNITY challenge raw data and process all sensor channels

    :param dataset: string
        Path with original OPPORTUNITY zip file
    :param output: string
        output path
    :param label: string, ["gestures", "locomotion", "both"(default)]
        Type of activities to be recognized. The OPPORTUNITY dataset includes several annotations to perform
        recognition modes of locomotion/postures and recognition of sporadic gestures.
    """

    data_dir = check_data(dataset)

    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    if label =='both':
        data_y = np.empty((0,2))
    else:
        data_y = np.empty((0,1))

    zf = zipfile.ZipFile(dataset)
    print ('Processing dataset files ...')
    for filename in OPPORTUNITY_TRAIN_DATA_FILES:
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print ('... file {0}'.format(filename))
            x, y = process_dataset_file(data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.vstack((data_y, y))
        except KeyError:
            print ('ERROR: Did not find {0} in zip file'.format(filename))

    X_train, y_train = data_x, data_y

    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    if label =='both':
        data_y = np.empty((0,2))
    else:
        data_y = np.empty((0,1))


    for filename in OPPORTUNITY_TEST_DATA_FILES:
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print ('... file {0}'.format(filename))
            x, y = process_dataset_file(data, label)
            data_x = np.vstack((data_x, x))
            data_y = np.vstack((data_y, y))
        except KeyError:
            print ('ERROR: Did not find {0} in zip file'.format(filename))
   
    X_test, y_test = data_x, data_y

    print ("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))

    # np.save('X_train_OPP', X_train)
    # np.save('y_train_OPP', y_train)
    # np.save('X_test_OPP', X_test)
    # np.save('y_test_OPP', y_test)

    window = 150
    stride = 75

    X_train, y_train = sliding_window(X_train, y_train[:,0], window, stride)
    X_test, y_test = sliding_window(X_test, y_test[:,0], window, stride)

    # Removing data with 0 labels
    xtrain = X_train[(y_train > 0)[:,0]]
    ytrain = y_train[y_train > 0] - 1
    xtest = X_test[(y_test > 0)[:,0]]
    ytest =  y_test[y_test > 0] - 1
    
    print ("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape,X_test.shape))
    print ("Final labels with size: | train {0} | test {1} | ".format(y_train.shape,y_test.shape))

    np.save(os.path.join(output, 'X_train_OPP.npy'), xtrain)
    np.save(os.path.join(output, 'y_train_OPP.npy'), ytrain)
    np.save(os.path.join(output, 'X_test_OPP.npy'), xtest)
    np.save(os.path.join(output, 'y_test_OPP.npy'), ytest)


def get_args():
    '''This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(
        description='Preprocess OPPORTUNITY dataset')
    # Add arguments
    parser.add_argument(
        '-i', '--input', type=str, help='OPPORTUNITY zip file', required=True)
    parser.add_argument(
        '-o', '--output', type=str, help='Output path', required=True)
    parser.add_argument(
        '-t', '--task', type=str.lower, help='Type of activities to be recognized', default="both", choices = ["gestures", "locomotion", "both"], required=False)
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    dataset = args.input
    target_filename = args.output
    label = args.task
    # Return all variable values
    return dataset, target_filename, label

if __name__ == '__main__':

    OpportunityUCIDataset_zip, output, l = get_args();
    generate_data(OpportunityUCIDataset_zip, output, l)

"""
d_name = 'OPP' #Enter dataset name

path = 'F:/Vikranth/Arm/Data'

X_train0 = np.load(os.path.join(path, 'X_train_{}.npy'.format(d_name)))
y_train_binary = np.load(os.path.join(path, 'y_train_{}.npy'.format(d_name)))
X_test0 = np.load(os.path.join(path,'X_test_{}.npy'.format(d_name)))
y_test_binary = np.load(os.path.join(path, 'y_test_{}.npy'.format(d_name)))

X_train0.shape

y_test_binary[:,0].shape

Counter(y_train_binary[:,0])

print(X_train0[4][0])

np.r_[0, 37:46, 50:59, 63:72, 76:85, 89:98, 243, 249].shape

X_train, y_train = sliding_window(X_train0,y_train_binary[:,0], 150, 75)
X_test, y_test = sliding_window(X_test0,y_test_binary[:,0], 150, 75)

Counter(y_test[:,0])

ytrain = y_train[y_train > 0] - 1
xtrain = X_train[(y_train > 0)[:,0]]
ytest =  y_test[y_test > 0] - 1
xtest = X_test[(y_test > 0)[:,0]]

Counter(ytrain)

print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

path2 = 'F:/Vikranth/Arm/Datasets/OPP'
d_name = 'OPP'
np.save(os.path.join(path2, 'X_train_{}.npy'.format(d_name)), xtrain)
np.save(os.path.join(path2, 'y_train_{}.npy'.format(d_name)), ytrain)
np.save(os.path.join(path2, 'X_test_{}.npy'.format(d_name)), xtest)
np.save(os.path.join(path2, 'y_test_{}.npy'.format(d_name)), ytest)
"""