"""
Module provides train functions to train transfer learning models
"""

import torch
import ssl
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def imshow(img):
    """
    show image
    Args:
        img(numpy array): image as numpy array
    Returns:
        -
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_augmentations():
    """
    show image augmentations
    Args:
        -
    Returns:
        -
    """
    images, labels = next(iter(train_loader))
    fig = plt.figure(figsize=(20, 10))
    for i in range(10):
        ax = fig.add_subplot(
            2, 5, i + 1, xticks=[], yticks=[], label=classes[labels[i]]
        )
        ax.set_title(classes[labels[i]])
        imshow(images[i])
    plt.show()


def train_model(model2train, train_epoch_number):
    """
    train the model
    Args:
        model2train(model): PyTorch model to train
        train_epoch_number(int): number of epochs to train
    Returns:
        -
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model2train.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(train_epoch_number):
        model2train.train()
        total_loss = 0.0

        # this for loop goes through each image and its label
        for images, labels in train_loader:
            # push images and labels onto the GPU
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # zero the parameter gradients for each new pic

            # for Inception
            #outputs, _ = model2train(

            # for EfficientNet
            outputs = model2train(
                images
            )  # "forward" the image through the model and get a prediction
            loss = criterion(
                outputs, labels
            )  # compare the prediction/ output with the real label to get the loss
            loss.backward()  # propagate loss backwards through whole net so all neurons can update
            optimizer.step()  # next step
            total_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{train_epoch_number}, Loss: {total_loss / len(train_loader)}"
            )  # print current loss, i.e. how good is the model in which episode?

    print("Training finished")

    return model2train


def validate_model(model, dataloader, criterion):
    """
    validate the model
    Args:
        model2train(model): PyTorch model to train
        dataloader(dataloader): PyTorch dataloader with image data
        criterion(criterion): PyTorch criterion to compute loss
    Returns:
        -
    """
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Forward pass

            loss = criterion(outputs, labels)  # Compute loss
            total_loss += loss.item()

            # Convert logits to probabilities and get predicted class
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predicted_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="weighted")
    recall = recall_score(all_labels, all_predictions, average="weighted")
    f1 = f1_score(all_labels, all_predictions, average="weighted")

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
    print(
        torch.cuda.is_available()
    )  # check if GPU is available, because they massively speed up training

    torch.cuda.empty_cache()  # clear GPU memory....

    # build the data transformer including image augmentations
    train_transforms = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.RandomRotation(10),  # rotate image 30 degrees
            transforms.ColorJitter(
                brightness=0.5,
                hue=0.3,
                contrast=0.5,
            ),  # modify contrast, brightness or hue with parameter as given probability
            transforms.RandomGrayscale(0.20),  # random grayscaling given probability
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_data = datasets.ImageFolder(
        "inputimages/train/", transform=train_transforms
    )  # load data from training directory in train_data dataloader
    validation_data = datasets.ImageFolder(
        "inputimages/val/", transform=validation_transform
    )
    test_data = datasets.ImageFolder("inputimages/test/", transform=test_transform)

    n_classes = len(
        train_data.classes
    )  # the 4 target classes: accelerate, decelerate, left, right
    # CHANGE JENS 15.01.2024
    #classes = ["accelerate", "decelerate", "left", "right"]
    classes = ["accelerate", "left", "right"]

    print("Number of training images: ", len(train_data))
    print("Number of validation images: ", len(validation_data))
    print("Number of test images: ", len(test_data))
    print("Number of target classes: ", n_classes)

    BATCH_SIZE = 32
    NUM_WORKERS = 2

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        validation_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True
    )

    MAX_EPOCH_NUMBER = 300

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ssl._create_default_https_context = ssl._create_unverified_context

    # Inception Transfer learning
    #transfer_learning_model = models.inception_v3()

    # Efficient Net
    transfer_learning_model = models.efficientnet_b0(weights ="EfficientNet_B0_Weights.DEFAULT")

    for param in transfer_learning_model.parameters():
        param.requires_grad = False  # Freeze all layers initially

    # Unfreeze specific layers if needed
    for child in transfer_learning_model.children():
        if isinstance(child, nn.Sequential):
            for param in child.parameters():
                param.requires_grad = True

    # Inception: Modify the classifier head for classification
    #num_ftrs = transfer_learning_model.fc.in_features
    #transfer_learning_model.fc = nn.Sequential(
    #    nn.Linear(num_ftrs, 256),
    #    nn.ReLU(),
    #    nn.Dropout(0.5),
    #    nn.Linear(
    #        256, n_classes
    #    ),  # Change the output size to match your number of classes
    #)

    # Efficient Net
    transfer_learning_model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes)

    transfer_learning_model.to(device)
    trained_transfer_learning_model = train_model(transfer_learning_model, MAX_EPOCH_NUMBER)

    criterion = nn.CrossEntropyLoss()

    # Assuming you have a trained model, a validation/test DataLoader, and a loss criterion
    (
        validation_loss,
        validation_accuracy,
        validation_precision,
        validation_recall,
        validation_f1,
    ) = validate_model(trained_transfer_learning_model, val_loader, criterion)

    torch.save(trained_transfer_learning_model, "models/TF_Learning.pth")
