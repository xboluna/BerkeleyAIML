from PIL import Image, ImageFile
import numpy as np
from typing import List, Tuple
import logging

import torch

import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable

from torchvision.models import resnet18


ImageFile.LOAD_TRUNCATED_IMAGES = True

# Label key
LABELS = {
        'neutral': 0,
        'sexy': 1,
        'porn': 2,
        #'drawing':3,
        #'hentai':4
        }


class CaptchaDataset(Dataset):
    def __init__(self, data, labels, transform = None):
        self.X = data
        self.y = labels
        self.T = transform
        return 

    def transform(self, image:np.ndarray) -> torch.Tensor:
        """Apply dataset transform."""
        if self.T is None:
            return T.Compose([
                        T.RandomRotation(30),
                        T.RandomResizedCrop(224),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        ])(image)
        else: 
            return self.T(image)

    def __getitem__(self, index:int) -> Tuple[torch.Tensor, str]:
        """Select one sample. DataLoader accesses samples through this function."""
        img = Image.open(self.X[index]).convert('RGB')
        return self.transform(img), self.y[index]

    def __len__(self) -> int:
        """Also needed for DataLoader."""
        return len(self.X)


class nnModuleWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = None
        self.CUDA = torch.cuda.is_available()

    def fit(self, trainloader:DataLoader,
            criterion = nn.MultiLabelSoftMarginLoss(),
            optimizer = torch.optim.Adam, 
            learning_rate:float = 1e-3,
            epochs:int = 20,
            testloader:DataLoader = None,
            ) -> None:
        """
        Run fitting process on our from a set of training images.

        Parameters:
        trainloader (DataLoader object) Training set
        criterion (nn Loss object) Loss criterion
        optimizer (Optimizer function) Optimizer function
        learning_rate (float) 
        epochs (int)
        testloader (DataLoader) optional, if you want to run val between epochs
        """

        # Check that we wrapped correctly
        model = self.network
        assert self.network is not None, 'self.network not defined!'

        # Instantiate optimizer
        optimizer = optimizer(model.parameters(), lr=learning_rate)

        logging.info(f'{len(trainloader)} batches per epoch.')


        ### OUTFILE
        print('Epoch, Train Loss, Val Loss')
        ####

        best_val_loss = None

        for epoch in range(epochs):
            
            logging.info(f'Starting epoch {epoch + 1}/{epochs}')
 
            running_loss = 0.
            for i, (images, label) in enumerate(trainloader):

                # Zero grads
                optimizer.zero_grad()

                # Forward iter
                prediction = model(images)

                # Calculate loss
                loss = criterion(
                        prediction.reshape( prediction.shape[0], len(LABELS) ),
                        label
                        )

                # Backpropagate
                loss.backward()

                # Step optimizer
                optimizer.step()

                # Log stats
                running_loss += loss.item() * images.size(0)
                logging.info(f'[{epoch + 1}, {i + 1}] loss: {loss:.3e}')
                
            logging.info(f'Finished epoch {epoch + 1}/{epochs} with total loss {running_loss}')


            if testloader is not None:
                _, val_loss = self.validate(testloader, criterion=criterion)


            ### OUTFILE
            print(f'{epoch + 1}, {running_loss}, {val_loss}')
            ####

            
            # Save if it's our best so far
            if best_val_loss is None:
                best_val_loss = val_loss
            elif val_loss <= best_val_loss:
                best_val_loss = val_loss
                torch.save(self.network.state_dict(), './ResNetWrapper_weights.pth')



        logging.info('Finished fitting.')
        return


    def validate(self, 
                 testloader:DataLoader,
                 criterion = None
                 ) -> Tuple[float]:
        """
        Validate the accuracy of our model from a set of validation images.

        Parameters: model
        testloader (DataLoader object)
        criterion -> if passed, will calculate loss
        """

        # Check that we wrapped correctly
        model = self.network
        assert self.network is not None, 'self.network not defined!'


        # Strings counter
        s_total = s_correct = 0
        fn_count = pred_pos_count = 0

        # If we keep track of loss
        if criterion is not None: running_loss = 0.

        # Not training -- don't need to calc. gradients
        with torch.no_grad():
            # For each test batch
            for (images, label) in testloader:

                # Predict label
                prediction = model(images)

                # For each label in batch
                for i,pred in enumerate(
                    prediction.reshape( prediction.shape[0], len(LABELS) )
                    ):

                    # Retrieve correct label
                    correct_label = label[i]

                    # Choose pred label with highest weight
                    predicted_label = np.argmax(pred)

                    # Do the captchas match?
                    if correct_label == predicted_label: s_correct += 1
                    s_total += 1

                    if correct_label == 2 or correct_label == 1:
                        pred_pos_count += 1
                        if predicted_label == 2 or predicted_label == 2: 
                            fn_count += 1 
                                    
                    # Log label comparison
                    logging.debug(f'Predicted: {predicted_label} -- Ground Truth: {correct_label}')

                # Increment loss
                if criterion is not None:
                    loss = criterion(
                            prediction.reshape( prediction.shape[0], len(LABELS) ),
                            label
                            )
                    running_loss += loss.item() * images.size(0)
                    
                # Log Accuracy
                logging.debug(f'Running Label Accuracy: {100 * s_correct/s_total:.1f}')
                logging.debug(f'Running number of False Negatives: {fn_count}')

        # Calculate results
        label_accuracy = s_correct / s_total
        logging.info(f'{len(testloader.dataset)} labels scanned')
        logging.info(f'Accuracy : {(100 * label_accuracy)}%')
        logging.info(f'Number of False Negatives: {fn_count}; as a pctg of all labels {100 * fn_count / pred_pos_count}')
        
        # Log loss
        if criterion is not None: logging.info(f'Epoch total validation loss {running_loss}')

        if criterion is None:
            return label_accuracy
        else:
            return label_accuracy, running_loss

class ResNetWrapper(nnModuleWrapper):
    # https://arxiv.org/pdf/1512.03385v1.pdf
    def __init__(self, model = None):
        super().__init__()

        if model is None: model = resnet18(pretrained = False)
        model.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        model.fc = nn.Linear(512, len(LABELS), # Five image classes: neutral, sexy, porn, drawing, hentai
                bias=True)

        self.network = model

    @classmethod
    def instantiate_with_no_weights(cls):
        return cls( model = resnet18() )

def debug_steps(input, net):
    output = input
    for step in net.children():
        output = step(output)
        print(step, output.shape)
    return output
