import numpy as np
import cv2
import os
import mss # library to make screenshots if multiple screens are available, i.e. laptop's monitor and another monitor.
import torch
from torchvision import transforms
import time
from getkeys import key_check

from directkeys import PressKey, ReleaseKey, W, A, S, D

def accelerate():
    print('accelerate')
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def decelerate():
    print('decelerate')
    PressKey(S)
    #ReleaseKey(W)
    #ReleaseKey(A)
    #ReleaseKey(D)
    ReleaseKey(S)

def left():
    print('driving left')
    #PressKey(W)
    PressKey(A)
    #time.sleep(0.1)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A)
    #ReleaseKey(S)

def right():
    print('driving right')
    #PressKey(W)
    PressKey(D)
    #time.sleep(0.1)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    #ReleaseKey()


def grab_screen(monitorNumber):
    """
    grab screen depending on monitor number and return in format 600x600 pixel as RGB via opencv2 library
    """
    with mss.mss() as sct:
        screen = np.array(sct.grab(sct.monitors[monitorNumber]))
        screen = cv2.resize(screen, (300, 300)) # resize in 300x300, reduces inference time
        screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB) # store as RGB, since MSS normally uses RGBA!!!!
    return screen

if __name__ == "__main__":

    print(torch.cuda.is_available()) #check if GPU is available, because they massively speed up training
    torch.cuda.empty_cache() # clear GPU memory....

    CNN_model = torch.load("models/model.pth")
    print("Model loaded")
    CNN_model.eval()

    device = next(CNN_model.parameters()).device

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    monitorNumber = 2

    with mss.mss() as sct:

        print("Driving automatically in 10 seconds.....prepare GTA 5")
        time.sleep(10)
        print("Ready to drive GTA 5!!")

        while (True):
            screen = grab_screen(monitorNumber)

            image_tensor = preprocess(screen).unsqueeze(0).to(device)


            with torch.no_grad():
                outputs = CNN_model(image_tensor[0])

                _, predicted_class_idx = outputs.max(1)

                direction = predicted_class_idx.item()
                print(direction)

                # todo: send keyboard press to GTA!!

                if direction == 0:
                    # press w
                    #print("up\n")
                    #PressKey(0x11)
                    #ReleaseKey(0x11)
                    accelerate()
                #elif direction == 1:
                    # press s
                    #print("down\n")
                    #PressKey(0x1F)
                    #ReleaseKey(0x1F)
                    #decelerate()
                elif direction == 2:
                    # press a
                    #print("left\n")
                    #PressKey(0x1E)
                    #ReleaseKey(0x1E)
                    left()
                elif direction == 3:
                    print("right\n")
                    # press d
                    #PressKey(0x20)
                    #ReleaseKey(0x20)
                    right()
