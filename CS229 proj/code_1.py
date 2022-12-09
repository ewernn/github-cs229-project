from PIL import Image
import numpy as np
import torch
from torchvision import datasets, models, transforms
from torchvision.models import resnet50, ResNet50_Weights, alexnet, AlexNet_Weights, squeezenet1_0, SqueezeNet1_0_Weights, vgg16, VGG16_Weights, densenet201, DenseNet201_Weights, inception_v3, Inception_V3_Weights
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch import nn
import time
import copy
import os
import shutil

def check_data():
    dataset = 'eric_split_data'
    for kind in os.listdir(dataset):
        if kind == '.DS_Store': continue
        kind_path = dataset+'/'+kind
        sum_kind = 0
        for country in os.listdir(kind_path):
            if country == '.DS_Store': continue
            f_country = kind_path+'/'+country
            sum_kind += len(os.listdir(f_country))
        print(f"{kind} data has {sum_kind} pics")


# model training code 
def debug_time(identifier,start):
    print(f"part {identifier} at {time.time()-start}")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def remove_dirs(file_path, directories_to_keep):
    for entry in os.listdir(file_path):
        # If the entry is a directory and it is not in the list of directories to keep
        # print(f"entry is {entry} and directories_to_keep is {directories_to_keep}")
        if os.path.isdir(entry) and entry not in directories_to_keep:

            # Recursively delete the directory and all its contents
            print(f"removing {entry} folder from {file_path} ")
            shutil.rmtree(entry)


def clean_directories(root_dir, selection_num):
    num_top_subfolders = selection_num
    train_dir = os.path.join(root_dir, 'train')

    subfolder_counts = {}

    # Iterate over the subfolders in the "train" directory
    for subfolder in os.listdir(train_dir):
        # Get the full path of the subfolder
        subfolder_path = os.path.join(train_dir, subfolder)

        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # Initialize the count of files in the subfolder to 0
            subfolder_counts[subfolder] = 0

            # Iterate over the files in the subfolder
            for file in os.listdir(subfolder_path):
                # Get the full path of the file
                file_path = os.path.join(subfolder_path, file)

                # Check if the file is a regular file
                if os.path.isfile(file_path):
                    # Increment the count of files in the subfolder
                    subfolder_counts[subfolder] += 1

    # Sort the subfolders by their counts of files
    sorted_subfolders = sorted(subfolder_counts, key=subfolder_counts.get, reverse=True)

    # Get the list of top subfolders
    top_subfolders = sorted_subfolders[:num_top_subfolders]

    for mode in ["train", "test", "val"]:
        remove_dirs(os.path.join(root_dir, mode), top_subfolders)


def load_data(data_path, model, feature_extract, input_size, batch_size):

    

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #double check these are for Resnet50
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
    }


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x), data_transforms[x]) for x in ['train', 'val']}
    

    print(f"my batch size is {batch_size}")
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=4
                                    ) 
        for x in ['train', 'val']
    }
    return dataloaders_dict


def params_to_learn(model, feature_extract):
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        num = 0
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                num +=1
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    print(f"num to update: {num}")
    return params_to_update

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, is_inception=False):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            start = time.time()
            # debug_time(0,start)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            # start_for = time.time()
            start = time.time()
            count = 0
            for inputs, labels in dataloaders[phase]:
                # print(f"label values: {labels}, \n label size: {np.shape(labels)}")
                # print(f"inputs size : {np.shape(inputs)}")
                print(f"iter: {count}")
                count+=1
                # debug_time(4,start)
                # start = time.time()
                # debug_time(1,start)
                # end_for = time.time()
                # print(f"{phase} for loop took {end_for - start_for} sec to complete")
                # start_for = time.time()
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # print(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # start_with = time.time()
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # end_with = time.time()
                # print(f"with thing took {end_with - start_with}")
                # debug_time(2,start)
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                end = time.time()
                #print(f"body run time: {end - start} sec")
                # debug_time(3,start)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            # debug_time(5,start)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            debug_time(6,start)
        print()
      
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



def run():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context


    check_data()
    model_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    weights_resnet = ResNet50_Weights.DEFAULT
    preprocess_resnet = weights_resnet.transforms() 

    model_names = ["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"]
    num_classes = 124 # CHECK
    # batch_sizes = [16, 32, 64, 128, 265]
    batch_size = 128
    num_epochs = 15
    feature_extract = True
    # Resnet50 takes inputs of dim (224,224,3)
    input_size = 224


    set_parameter_requires_grad(model_resnet, feature_extract)


    num_resnet = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_resnet, num_classes)

    # Feature Extraction Sanity Check
    params_to_update = params_to_learn(model_resnet, feature_extract)

    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    ################# GPU ##################
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"device: {device}")
    device = torch.device("mps")
    model_resnet = model_resnet.to(device)
    ################# GPU ##################


    data_path = 'top20_split_data'

    data_loader = load_data(data_path, model_resnet, feature_extract, input_size, batch_size)

    # Train  model
    model, hist = train_model(model_resnet, data_loader, criterion, optimizer, num_epochs, device)
    

if __name__ == '__main__':
    # clean_directories(data_path, 75)
    # freeze_support()
    run()
    # path = 'eric_split_data'
    # for mode in ["train", "val", "test"]:
        # print(len(os.listdir(os.path.join(path, mode))))
    print(torch.backends.mps.is_available())








