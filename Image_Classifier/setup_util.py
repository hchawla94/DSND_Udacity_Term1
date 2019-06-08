import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import PIL
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict

import argparse
import os

def load_data(data_dir = "./flowers"):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms ={'train':transforms.Compose([transforms.RandomRotation(40),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])]),
                      'test' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                           std=[0.229, 0.224, 0.225])]),
                      'valid' : transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                           std=[0.229, 0.224, 0.225])])}
    
    image_datasets = {'train' : torchvision.datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'test' : torchvision.datasets.ImageFolder(test_dir, transform=data_transforms['test']),
                      'valid' : torchvision.datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])}
    
    dataloaders = {'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True),
                   'valid' : torch.utils.data.DataLoader(image_datasets['valid'],batch_size=64, shuffle=True)} 
    
    dataset_sizes = {x: len(image_datasets[x]) 
                              for x in ['train', 'valid', 'test']}
    
    return dataloaders, dataset_sizes, image_datasets

def model_setup(network_name ='vgg16',dropout=0.6,hidden_layer_1=1500, hidden_layer_2 = 1000, hidden_layer_3 = 500, power='gpu'):
    #Load a pre-trained model. VGG16 as default
    if network_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    elif network_name == "densenet161":
        model = models.densenet161(pretrained=True)
        model.name = "densenet161"
    elif network_name == "alexnet":
        model = models.alexnet(pretrained=True)
        model.name = "alexnet"
    else:
        print("Only accepts vgg16, densenet161, or alexnet")

    # Freezing the weights of the pretrained model per recommendation such that we don't end up updating them by backpropping
    for param in model.parameters():
        param.requires_grad = False
    
    input_size = model.classifier[0].in_features
    output_size = 102
    
    if network_name == "densenet161":
        model.classifier = nn.Sequential(OrderedDict([
                                  ('input', nn.Linear(input_size, hidden_layer_1)),
                                  ('dropout1', nn.Dropout(p=dropout)),
                                  ('relu1', nn.ReLU()),
                                  ('hidden_layer_1', nn.Linear(hidden_layer_1 , hidden_layer_2)),
                                  ('dropout2', nn.Dropout(p=dropout)),
                                  ('relu2', nn.ReLU()),
                                  ('hidden_layer_2', nn.Linear(hidden_layer_2, hidden_layer_3)),
                                  ('relu3', nn.ReLU()),
                                  ('hidden_layer_3', nn.Linear(hidden_layer_3, output_size)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
    elif network_name == 'vgg16':
        model.classifier = nn.Sequential(OrderedDict([
                                  ('input', nn.Linear(input_size, hidden_layer_1)),
                                  ('dropout1', nn.Dropout(p=dropout)),
                                  ('relu1', nn.ReLU()),
                                  ('hidden_layer_1', nn.Linear(hidden_layer_1, hidden_layer_2)),
                                  ('dropout2', nn.Dropout(p=dropout)),
                                  ('relu2', nn.ReLU()),
                                  ('hidden_layer_2', nn.Linear(hidden_layer_2, hidden_layer_3)),
                                  ('relu3', nn.ReLU()),
                                  ('hidden_layer_3', nn.Linear(hidden_layer_3, output_size)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
    elif network_name == 'alexnet':
        model.classifier = nn.Sequential(OrderedDict([
                                  ('input', nn.Linear(input_size, hidden_layer_1)),
                                  ('dropout1', nn.Dropout(p=dropout)),
                                  ('relu1', nn.ReLU()),
                                  ('hidden_layer_1', nn.Linear(hidden_layer_1, hidden_layer_2)),
                                  ('dropout2', nn.Dropout(p=dropout)),
                                  ('relu2', nn.ReLU()),
                                  ('hidden_layer_2', nn.Linear(hidden_layer_2, hidden_layer_3)),
                                  ('relu3', nn.ReLU()),
                                  ('hidden_layer_3', nn.Linear(hidden_layer_3, output_size)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
    else:
        print("Only accepts vgg16, densenet161, or alexnet")
    
    if torch.cuda.is_available() and power == 'gpu':
        model.cuda()
      
    return model

def train_model(model, dataloaders, dataset_sizes, image_datasets, criterion, optimizer, scheduler,num_epochs=15, power='gpu'):
    import torch.cuda
    if torch.cuda.is_available() and power == 'gpu':
        device='cuda'
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                import torch
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def model_params(model, lr=0.001, epochs=15):
    # Criterion- NLLLoss (recommended with Softmax final layer)
    criterion = nn.NLLLoss()
    # Adam Optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    # Decay LR by a factor of 0.1 every 4 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # Epochs
    eps=epochs
    
    return criterion, optimizer, scheduler, eps

def save_checkpoint(model, image_datasets, path='checkpoint.pth', network_name = 'vgg16', hidden_layer_1 = 1500, hidden_layer_2 = 1000, hidden_layer_3 = 500):
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    checkpoint = {'network_name': model.name,
                  'classifier': model.classifier,
                  'hidden_layer_1': hidden_layer_1,
                  'hidden_layer_2': hidden_layer_2,
                  'hidden_layer_3': hidden_layer_3,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}
    
    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(path='checkpoint.pth'):
    # Load the saved file
    checkpoint = torch.load(path)
    
    network_name = checkpoint['network_name']
    hidden_layer_1 = checkpoint['hidden_layer_1']
    hidden_layer_2 = checkpoint['hidden_layer_2']
    hidden_layer_3 = checkpoint['hidden_layer_3']
    model = model_setup(network_name, 0.6, hidden_layer_1, hidden_layer_2, hidden_layer_3)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image):
    
    image_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
   # image=os.path.join(image)
    PIL_image = PIL.Image.open(image)
    PIL_image = image_loader(PIL_image).float()
    
    np_image = np.array(PIL_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std_dev    
    np_image = np.transpose(np_image, (2, 0, 1))
            
    return np_image

def predict(cat_to_name, image_path, model, top_k=5):
    # Can set model to CPU, no need for GPU as such
    model.to("cpu")
    
    # Setting model to evaluate
    model.eval();
    
    # Process image and convert it from numpy to tensor
    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor).to("cpu").unsqueeze(0)
    model_input = image_tensor
    
    # Find probabilities by passing through the function and convert the prob to linear scale from log scale
    linear_probs = torch.exp(model.forward(model_input))
    
    # Find Top probs and detach the details
    top_probs, top_labels = linear_probs.topk(top_k)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labels = top_labels.detach().numpy().tolist()[0]
    
     # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers
