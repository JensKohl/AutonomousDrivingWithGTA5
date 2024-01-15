"""
Module providing a function to drive the vehicle.
Grabs an image, feeds image to CNN-model,
finds out direction for driving,
and steers vehicle via key press.
"""
import time
import numpy as np
import cv2
import mss  # make screenshots if multiple screens are available, i.e. laptop and another monitor.
import torch
from torchvision import transforms
import keyboard
from driving_commands import accelerate, steer_left, steer_right #, decelerate currently not used

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


if __name__ == "__main__":
    print(
        torch.cuda.is_available()
    )  # check if GPU is available, because they massively speed up training
    torch.cuda.empty_cache()  # clear GPU memory....

    CNN_model = torch.load("models/model.pth")
    print("Model loaded")
    CNN_model.eval()

    device = next(CNN_model.parameters()).device

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
                print("You pressed 'x'. Quitting....")
                break

            with torch.no_grad():
                outputs = CNN_model(image_tensor[0])

                _, predicted_class_idx = outputs.max(1)

                direction = predicted_class_idx.item()
                print(direction)

                # NOTE: deceleration not used at the moment
                # accelerate
                if direction == 0:
                    # print("up\n")
                    # PressKey(0x11)
                    # ReleaseKey(0x11)
                    accelerate()

                # decelerate, currently not used
                # elif direction == 1:
                # press s
                # print("down\n")
                # PressKey(0x1F)
                # ReleaseKey(0x1F)
                # decelerate()
                # steer left. if decelerate commented out, this is 2
                elif direction == 1: # steer left;
                    # press a
                    # print("left\n")
                    # PressKey(0x1E)
                    # ReleaseKey(0x1E)
                    steer_left()

                # steer right. if decelerate commented out, this is 3
                elif direction == 3:
                    print("right\n")
                    # press d
                    # PressKey(0x20)
                    # ReleaseKey(0x20)
                    steer_right()
