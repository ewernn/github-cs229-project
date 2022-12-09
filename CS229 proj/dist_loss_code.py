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
import geopy

### GEOGRAPHIC DISTANCE CALCULATION ###
def calc_dist_matrix(normalize=True):
    coord_data = 'world_country_and_usa_states_latitude_and_longitude_values.csv'
    dataset = 'top20_split_data'

    ### get country coordinates ###
    my_data_str = np.genfromtxt(coord_data, delimiter=',', skip_header=1, dtype=str)
    lats = [float(x) for x in my_data_str[:,1]]
    longs = [float(x) for x in my_data_str[:,2]]
    countries_with_coords = my_data_str[:,3]
    coords = dict()
    for i in range(len(countries_with_coords)):
        coords[countries_with_coords[i]] = (lats[i],longs[i])

    ### get list of countries from dataset ###
    test_dir = os.listdir(dataset)[0]
    if test_dir == '.DS_Store': test_dir = os.listdir(dataset)[1]
    data_countries = os.listdir(dataset+'/'+test_dir)
    if '.DS_Store' in data_countries: data_countries.remove('.DS_Store')
    # print(f"data_countries: {data_countries}")
    ### calculate distance matrix ###
    n_countries = len(data_countries)
    from geopy import distance
    dist_matrix = np.zeros((n_countries,n_countries))
    ### calculate matrix ###
    for a,cunt_a in enumerate(data_countries):
        for b,cunt_b in enumerate(data_countries):
            cunt_a,cunt_b = cunt_a[4:],cunt_b[4:]  # remove prefix 'new_'
            if coords.get(cunt_a) is None or coords.get(cunt_b) is None: continue
            dist_matrix[a,b] = distance.great_circle(coords[cunt_a], coords[cunt_b]).miles
    if not normalize: return dist_matrix
    if np.max(dist_matrix) == 0: print(f"\n\n {dist_matrix} \n\n ERROR!! \n\n")
    return dist_matrix / np.max(dist_matrix)  # normalize

# compute loss_distance
loss_dist_matrix = calc_dist_matrix()  # shape=(n_countries,n_countries)
def compute_loss_dist(preds,labels,batch_size):
    loss = 0
    for i in range(batch_size):
        loss += loss_dist_matrix[preds[i],labels[i]]
    return loss

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

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

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, alpha, is_inception=False):
    since = time.time()
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # loss_dist_matrix = dist_matrix()  # shape=(n_countries,n_countries)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            count = 0
            last_time = time.time()
            print(f"normalize with: {len(dataloaders[phase].dataset)}")
            for inputs, labels in dataloaders[phase]:
                if count % 295 == 0: print(f"iter: {count}.     time for last iter: {time.time()-last_time}")
                last_time = time.time()
                count+=1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    loss_dist = compute_loss_dist(preds, labels, inputs.size(0))
                    #print(f"loss = {loss}.  loss_dist = {loss_dist}")
                    loss += alpha * loss_dist


                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        print()
      
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #########################################
    ######### UNCOMMENT LATER ###############
    # model.load_state_dict(best_model_wts)
    #########################################
    return model, val_acc_history


def init_resnet():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    model_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_classes = 20 # CHECK
    feature_extract = True

    set_parameter_requires_grad(model_resnet, feature_extract)
    num_resnet = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_resnet, num_classes)

    ################# GPU ##################
    device = torch.device("mps")
    model_resnet = model_resnet.to(device)
    ################# GPU ##################

    return model_resnet
    

def run():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    model_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    weights_resnet = ResNet50_Weights.DEFAULT
    preprocess_resnet = weights_resnet.transforms() 

    model_names = ["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"]
    num_classes = 20 # CHECK
    # batch_sizes = [16, 32, 64, 128, 265]
    batch_size = 128
    num_epochs = 15
    feature_extract = True
    alpha = .25
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
    device = torch.device("mps")
    model_resnet = model_resnet.to(device)
    ################# GPU ##################

    data_path = 'top20_split_data'
    data_loader = load_data(data_path, model_resnet, feature_extract, input_size, batch_size)

    # Train  model

    # set by eric
    num_epochs = 1
    alphas = [0,.1,.25,.5,.75,1.25]
    histories = np.zeros((len(alphas),num_epochs))
    for i,alpha in enumerate(alphas):
        model, hist = train_model(model_resnet, data_loader, criterion, optimizer, num_epochs, device, alpha)
        model_reset = init_resnet()
        # histories[i,:len(hist)] = hist
    print(f"final accuracies for alphas = {alphas}:\n {histories}")
    # model, hist = train_model(model_resnet, data_loader, criterion, optimizer, num_epochs, device, alpha)
    

if __name__ == '__main__':
    assert torch.backends.mps.is_available() == True
    run()








