import torch
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
import splitfolders
from torch.optim import Adam
import torch.nn as nn
from model import CNNModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def show_augmentations():
    images, labels = next(iter(train_loader))
    fig = plt.figure(figsize=(20, 10))
    for i in range(10):
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[], label=classes[labels[i]])
        ax.set_title(classes[labels[i]])
        imshow(images[i])
    plt.show()


def train_model(model2train, epoch_number):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model2train.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(epoch_number):
        model2train.train()
        total_loss = 0.0

        # this for loop goes through each image and its label
        for images, labels in train_loader:
            # push images and labels onto the GPU
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # zero the parameter gradients for each new pic
            outputs, _ = model2train(images) # "forward" the image through the model and get a prediction
            loss = criterion(outputs, labels) # compare the prediction/ output with the real label to get the loss
            loss.backward() # propagate the loss backwards, that is through the whole net so all involved neurons can update
            optimizer.step() # next step
            total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{max_epoch_number}, Loss: {total_loss / len(train_loader)}") # print current loss, i.e. how good is the model in which episode?

    print("Training finished")

    return model2train


def validate_model(model, dataloader, criterion):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs) # Forward pass

            loss = criterion(outputs, labels) # Compute loss
            total_loss += loss.item()

            # Convert logits to probabilities and get predicted class
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predicted_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Validation/Test Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    # Confusion Matrix
    confusion = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(confusion)

    return loss, accuracy, precision, recall, f1

if __name__ == "__main__":
    print(torch.cuda.is_available()) #check if GPU is available, because they massively speed up training

    torch.cuda.empty_cache() # clear GPU memory....

    # build the data transformer including image augmentations
    train_transforms = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.RandomRotation(30), #rotate image 30 degrees
        transforms.ColorJitter(brightness=.5, hue=.3, contrast=.5, ), #modify contrast, brightness or hue with parameter as given probability
        transforms.RandomGrayscale(.20), # random grayscaling given probability
        transforms.ToTensor(), # convert to tensor
        transforms.Normalize(mean = [0.5, 0.5, 0.5],
                             std = [0.5, 0.5, 0.5])])

    validation_transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(), # convert to tensor
        transforms.Normalize(mean = [0.5, 0.5, 0.5],
                             std = [0.5, 0.5, 0.5])])

    test_transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(), # convert to tensor
        transforms.Normalize(mean = [0.5, 0.5, 0.5],
                             std = [0.5, 0.5, 0.5])])

    train_data = datasets.ImageFolder("inputimages/train/", transform=train_transforms)  # load data from training directory in train_data dataloader
    validation_data = datasets.ImageFolder("inputimages/val/", transform=validation_transform)
    test_data = datasets.ImageFolder("inputimages/test/", transform=test_transform)

    n_classes = len(train_data.classes) # the 4 target classes: accelerate, decelerate, left, right
    classes = ["accelerate", "decelerate", "left", "right"]

    print('Number of training images: ', len(train_data))
    print('Number of validation images: ', len(validation_data))
    print('Number of test images: ', len(test_data))
    print('Number of target classes: ', n_classes)

    batch_size = 128  # normally we should take a batch size higher than 32, e.g. 64...but Colab crashed regularly with more than 32.
    num_workers=2

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = num_workers, shuffle = True)
    val_loader = torch.utils.data.DataLoader(validation_data, batch_size = batch_size, num_workers = num_workers, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = num_workers, shuffle = True)

    max_epoch_number = 300

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Inception_Model = models.inception_v3()

    for param in Inception_Model.parameters():
        param.requires_grad = False  # Freeze all layers initially

    # Unfreeze specific layers if needed
    for child in Inception_Model.children():
        if isinstance(child, nn.Sequential):
            for param in child.parameters():
                param.requires_grad = True

    # Modify the classifier head for binary classification
    num_ftrs = Inception_Model.fc.in_features

    Inception_Model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, n_classes)  # Change the output size to match your number of classes
    )

    Inception_Model.to(device)
    trained_inception_model = train_model(Inception_Model, max_epoch_number)

    criterion = nn.CrossEntropyLoss()

    # Assuming you have a trained model, a validation/test DataLoader, and a loss criterion
    validation_loss, validation_accuracy, validation_precision, validation_recall, validation_f1 = validate_model(trained_inception_model, val_loader, criterion)

    torch.save(trained_inception_model, 'models/Inception.pth')

