"""
This module grab screens in GTA Vice City when the vehicle moves..
"""

import time
from datetime import datetime
import numpy as np
import cv2
# we need this library to monitor if specific keys are pressed.
# In our case "w" and "s" for accelerating or braking and "a" and "d" for steering -
# and of course their combination, e.g. "wa", "wd", "sa"
from pynput import (
    keyboard,
)
import mss  # make screenshots if multiple screens are available, i.e. laptop and another monitor.

def test_screenshots():
    """
    test screenshot function
    Args:
        -
    Returns:
        -
    """
    print(sct.monitors)  # gives us coordinates of all monitor
    sct.shot(mon=-1, output="Monitor1and2.png")  # screenshot of monitor 1
    sct.shot(mon=1, output="Monitor1.png")  # screenshot of monitor 1
    sct.shot(mon=2, output="Monitor2.png")  # screenshot of monitor 2


def grab_screen(monitor_number):
    """
    grab screen depending on monitor number and return
    in format 600x600 pixel in RGB channels via opencv2 library
    Args:
        monitor_number(int): which monitor
    Returns:
        screen: image
    """
    with mss.mss() as sct:
        screen = np.array(sct.grab(sct.monitors[monitor_number]))
        screen = cv2.resize(screen, (300, 300))
        screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)  # change jens 01.11.2023
    return screen


def on_press(key):
    """
    depending on specific key pressed, make a screenshot and store in different directory
    Args:
        key(string): which key was pressed
    Returns:
        boolean value
    """
    with mss.mss():
        monitor_number = 2

        if hasattr(key, "char"):
            current_date = datetime.utcnow()
            datestring = (
                str(current_date.year)
                + str(current_date.month)
                + str(current_date.day)
                + str(current_date.hour)
                + str(current_date.minute)
                + str(current_date.second)
                + ".jpg"
            )

            if key.char == ("w"):  # accelerate
                file_name = "images/accelerate/" + datestring
                screen = grab_screen(monitor_number)
                cv2.imwrite(file_name, screen)
                return True

            if key.char == ("s"):  # brake
                file_name = "images/decelerate/" + datestring
                screen = grab_screen(monitor_number)
                cv2.imwrite(file_name, screen)
                return True

            if key.char == ("a"):  # steer left
                file_name = "images/left/" + datestring
                screen = grab_screen(monitor_number)
                cv2.imwrite(file_name, screen)
                return True

            if key.char == ("d"):  # steer left
                file_name = "images/right/" + datestring
                screen = grab_screen(monitor_number)
                cv2.imwrite(file_name, screen)
                return True

            if key.char == "q":  # quit
                return False

        if key == keyboard.Key.esc:
            return False

def on_release(key):
    """
    checks which key was released
    Args:
        key(string): which key was released
    Returns:
        boolean value
    """
    if hasattr(key, "char"):
        if key.char == "q":
            # Stop listener
            return False
    if key == keyboard.Key.esc:
        return False
    return True


# Main function
if __name__ == "__main__":
    with mss.mss() as sct:
        print("Screens are recorded on pressing w,a,s,d. Press q for quit")
        time.sleep(5)
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
