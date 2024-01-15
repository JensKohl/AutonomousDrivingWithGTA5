"""
Module providing a function to drive the vehicle.
Grabs an image, feeds image to CNN-model,
finds out direction for driving,
and steers vehicle via key press.
"""

import time
import numpy as np
import cv2
import mss  # library to make screenshots of multiple screens, i.e. laptop and another monitor.
import torch
from torchvision import transforms
import keyboard
#from get_training_data import grab_screen
from driving_commands import accelerate, steer_left, steer_right #, decelerate


def grab_screen(monitor_number):
    """grab screen depending on monitor number and
        return image as 600x600 RGB pixel via opencv2 library
    Args:
        monitor_number (int): number of monitor (0 = all monitors, 1 = laptop, 2 = monitor)
    Returns:
        screen: image screen
    """
    with mss.mss() as sct:
        screen = np.array(sct.grab(sct.monitors[monitor_number]))
        screen = cv2.resize(
            screen, (300, 300)
        )  # resize in 300x300, reduces inference time
        screen = cv2.cvtColor(
            screen, cv2.COLOR_RGBA2RGB
        )  # store as RGB, since MSS normally uses RGBA!!!!
    return screen

def preprocess_image(image):
    """preprocess image_old
    Args:
        image(np.array): image as numpy array
    Returns:
        image: modified image
    """
    image = cv2.resize(image, (300, 300))

    # Convert the image to float32 and normalize
    image = image.astype(np.float32) / 255.0

    print(image.shape)
    # image -= np.array([0.485, 0.456, 0.406])
    # image /= np.array([0.229, 0.224, 0.225])

    return image

if __name__ == "__main__":
    print(
        torch.cuda.is_available()
    )  # check if GPU is available, because they massively speed up training
    torch.cuda.empty_cache()  # clear GPU memory....

    tf_model = torch.load("models/TF_Learning.pth")
    print("Model loaded")
    tf_model.eval()

    device = next(tf_model.parameters()).device

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    SELECTED_MONITOR_NUMBER = 2

    with mss.mss() as sct:
        print("Driving automatically in 10 seconds.....prepare GTA 5")
        time.sleep(10)
        print("Ready to drive GTA 5!!")

        while True:
            screen = grab_screen(SELECTED_MONITOR_NUMBER)

            image_tensor = preprocess(screen).unsqueeze(0).to(device)

            if keyboard.is_pressed("x"):
                print("You pressed 'x'. Quitting self-driving....")
                break

            with torch.no_grad():
                outputs = tf_model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_direction = torch.argmax(probabilities, dim=1).item()
                #print(predicted_direction)

                # NOTE: deceleration not used at the moment
                if predicted_direction == 0:
                    # press w
                    # print("up\n")
                    # PressKey(0x11)
                    # ReleaseKey(0x11)
                    accelerate()
                # elif direction == 1:
                # press s
                # print("down\n")
                # PressKey(0x1F)
                # ReleaseKey(0x1F)
                # decelerate()
                elif predicted_direction == 1:
                    # press a
                    # print("left\n")
                    # PressKey(0x1E)
                    # ReleaseKey(0x1E)
                    steer_left()
                elif predicted_direction == 2:
                    print("right\n")
                    # press d
                    # PressKey(0x20)
                    # ReleaseKey(0x20)
                    steer_right()
