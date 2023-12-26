import numpy as np
import cv2
import time
from pynput import keyboard # we need this library to monitor if specific keys are pressed. In our case "w" and "s" for accelerating or braking and "a" and "d" for steering - and of course their combination, e.g. "wa", "wd", "sa"
import os
import mss # library to make screenshots if multiple screens are available, i.e. laptop's monitor and another monitor.
import torch
from test_model import process_image, test_model
from get_training_data import grab_screen
from torchvision import transforms

def preprocess_image(image):
    image = cv2.resize(image, (300, 300))

    # Convert the image to float32 and normalize
    image = image.astype(np.float32) / 255.0

    print(image.shape)
    #image -= np.array([0.485, 0.456, 0.406])
    #image /= np.array([0.229, 0.224, 0.225])

    return image


def test_single_image(model, image):
    device = next(model.parameters()).device

    img_transforms = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225] )
    ])

    img = img_transforms(image)
    img_tensor = torch.unsqueeze(img, 0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor) # Forward pass
        # Convert logits to probabilities and get predicted class
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class


def predict_drive_direction(model, preprocessed_image):
    device = next(model.parameters()).device
    image = preprocess_image(preprocessed_image)

    # Convert the preprocessed image to a PyTorch tensor. TODO: failure here!!
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
   

    with torch.no_grad():
        outputs = model(image_tensor) # Forward pass
        # Convert logits to probabilities and get predicted class
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class


def process_image(filepath):
    image = cv2.imread(filepath)

    # Resize the image to the model's input size
    image = cv2.resize(image, (300, 300))

    # Convert the image to float32 and normalize
    image = image.astype(np.float32) / 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])

    return image


def test_single_image_test(model, image):
    model.eval()
    device = next(model.parameters()).device

    image = cv2.resize(image, (300, 300))
    image = image.astype(np.float32) / 255.0
    #image -= np.array([0.485, 0.456, 0.406])
    #image /= np.array([0.229, 0.224, 0.225])


    # Convert the preprocessed image to a PyTorch tensor
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor) # Forward pass
        # Convert logits to probabilities and get predicted class
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class


# todo: rewrite as soon as drive_vehicle_CNN.py works....

if __name__ == "__main__":

    print(torch.cuda.is_available()) #check if GPU is available, because they massively speed up training
    torch.cuda.empty_cache() # clear GPU memory....

    Inception_model = torch.load("models/Inception.pth")
    print("Model loaded")
    Inception_model.eval()

    monitorNumber = 2

    keyboard = keyboard.Controller()

    with mss.mss() as sct:

        while (True):

            print("Driving automatically in 5 seconds.....prepare GTA 5")
            time.sleep(5)

            screen = grab_screen(monitorNumber)
            print(screen.shape)

            test_single_image_test(Inception_model, screen)

            if direction == 0:
                keyboard.press('w')
                keyboard.release('w')
            elif direction == 1:
                keyboard.press('s')
                keyboard.release('s')
            elif direction == 2:
                keyboard.press('a')
                keyboard.release('a')
            elif direction == 3:
                keyboard.press('d')
                keyboard.release('d')
