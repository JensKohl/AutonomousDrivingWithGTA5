from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn as nn
import cv2

def test_model(model, test_loader, criterion):
    """
    Test a PyTorch model on a test dataset and compute evaluation metrics.
    Args:
        model (nn.Module): The PyTorch model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion: The loss criterion for the task.

    Returns:
        accuracy (float): Accuracy of the model on the test dataset.
        precision (float): Precision of the model's predictions.
        recall (float): Recall of the model's predictions.
        f1 (float): F1-score of the model's predictions.
    """
    model.eval()
    device = next(model.parameters()).device
    all_predictions = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Forward pass

            loss = criterion(outputs, labels) # Compute loss
            total_loss += loss.item() # and add up

            probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities and get predicted class
            predicted_class = torch.argmax(probabilities, dim=1) # from predictions take the one with the highest value

            all_predictions.extend(predicted_class.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Test Loss: {total_loss / len(test_loader)}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    # Confusion Matrix
    confusion = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(confusion)

    return accuracy, precision, recall, f1

def process_image(filepath):
    image = cv2.imread(filepath)

    # Resize the image to the model's input size
    image = cv2.resize(image, (300, 300))

    # Convert the image to float32 and normalize
    image = image.astype(np.float32) / 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])

    return image

def test_single_image(model, image_path):
    model.eval()
    device = next(model.parameters()).device
    image = process_image(image_path)

    # Convert the preprocessed image to a PyTorch tensor
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor) # Forward pass
        # Convert logits to probabilities and get predicted class
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

if __name__ == "__main__":

    print(torch.cuda.is_available()) #check if GPU is available, because they massively speed up training
    torch.cuda.empty_cache() # clear GPU memory....

    test_transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(), # convert to tensorcl
        transforms.Normalize(mean = [0.5, 0.5, 0.5],
                             std = [0.5, 0.5, 0.5])])

    batch_size = 64  # normally we should take a batch size higher than 32, e.g. 64...but Colab crashed regularly with more than 32.
    num_workers=2

    test_data = datasets.ImageFolder("inputimages/test/", transform=test_transform)
    n_classes = len(test_data.classes) # the 4 target classes: accelerate, decelerate, left, right
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = num_workers, shuffle = True)

    print('Number of test images: ', len(test_data))

    trained_model = torch.load("models/model.pth")
    trained_model.eval()

    criterion = nn.CrossEntropyLoss()

    test_accuracy, test_precision, test_recall, test_f1 = test_model(trained_model, test_loader, criterion)

    Inception_model = torch.load("models/Inception.pth")
    Inception_model.eval()

    test_accuracy, test_precision, test_recall, test_f1 = test_model(Inception_model, test_loader, criterion)

    testfile = "inputimages/test/accelerate/202311110439.jpg"

    class_labels = ["accelerate", "decelerate", "left", "right"]

    prediction = test_single_image(Inception_model, testfile)
    print("PREDICTION:{} has result: {}".format(testfile, class_labels[prediction]))
