### Fit params

DATA_DIR = './datascraper/data/train/',

BATCH_SIZE = 128
TEST_SIZE = .2
NUM_WORKERS = 4 

EPOCHS = 20
LEARNING_RATE = 1e-3

###
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, filename="./logs/run_fitting.log")

from sys import getsizeof

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder

from sklearn.model_selection import train_test_split

from Infrastructure import CaptchaDataset, ResNetWrapper


def run():

    # Label key
    LABELS = {
            'neutral': 0,
            'sexy': 1,
            'porn': 2,
            'drawing':3,
            'hentai':4
            }
    
    # Location of image directories
    directories = {
            'neutral': '%s/neutral/'%DATA_DIR,
            'sexy': '%s/sexy/'%DATA_DIR,
            'porn': '%s/porn/'%DATA_DIR,
            #'drawing': '%s/drawings/'%DATA_DIR,
            #'hentai':'%s/hentai/'%DATA_DIR,
            }


    for label in directories:

        # Collect the files
        file_locations = [ img for img in Path(directories[label]).glob('*') ]

        # Just reassign directories vals to the glob
        directories[label] = file_locations

    
    X = [ img for img in directories[label] for label in directories]
    y = [ encode for encode in [LABELS[label]]*len(directories[label]) for label in directories ]

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = TEST_SIZE )
    X_test, X_val, y_test, y_val = train_test_split(X,y, test_size = 0.3)
    
    # Locate files
    trainloader = DataLoader(CaptchaDataset(X_train,y_train), BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
    testloader = DataLoader(CaptchaDataset(X_test,y_test), BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
    valloader = DataLoader(CaptchaDataset(X_val, y_val), BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)

    logging.info( f'Identified {len(X)} images.' )
    
    net = ResNetWrapper()

    net.fit(trainloader, 
            criterion = nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam,
            learning_rate = LEARNING_RATE,
            epochs = EPOCHS,
            testloader = testloader)

    
    # Validating accuracy of model
    logging.info(f'Verifying accuracy')
    
    label_accuracy, char_accuracy = net.validate(valloader)
    

    logging.info('run_fitting.py complete.')
    return

if __name__ == "__main__":
    run()

