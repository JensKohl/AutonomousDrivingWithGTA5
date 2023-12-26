# This file has functions to grab screens in GTA Vice City when the vehicle moves.

import numpy as np
import cv2
import time
from pynput import keyboard # we need this library to monitor if specific keys are pressed. In our case "w" and "s" for accelerating or braking and "a" and "d" for steering - and of course their combination, e.g. "wa", "wd", "sa"
import os
import mss # library to make screenshots if multiple screens are available, i.e. laptop's monitor and another monitor.
from datetime import datetime



def test_screenshots():
    print(sct.monitors) #gives us coordinates of all monitor
    sct.shot(mon = -1, output ="Monitor1and2.png") # screenshot of monitor 1
    sct.shot(mon = 1, output ="Monitor1.png") # screenshot of monitor 1
    sct.shot(mon = 2, output ="Monitor2.png") # screenshot of monitor 2

def grab_screen(monitorNumber):
    """
    grab screen depending on monitor number and return in format 600x600 pixel via opencv2 library
    """
    with mss.mss() as sct:
        screen = np.array(sct.grab(sct.monitors[monitorNumber]))
        screen = cv2.resize(screen, (300, 300)) 
        screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB) # change jens 01.11.2023
    return screen
 
def on_press(key):
    """
    depending on specific key pressed, make a screenshot and store in different directory
    """
    with mss.mss() as sct:
        if hasattr(key, 'char'):
            current_date = datetime.utcnow()
            datestring = str(current_date.year) + str(current_date.month) + str(current_date.day) + str(current_date.hour) + str(current_date.minute) + str(current_date.second) + ".jpg"
            monitorNumber = 2

            if key.char == ('w'): #accelerate
                FileName = "images/accelerate/" + datestring
                screen = grab_screen(monitorNumber)
                cv2.imwrite(FileName, screen)

            elif key.char == ('s'): # brake
                FileName = "images/decelerate/" + datestring
                screen = grab_screen(monitorNumber)
                cv2.imwrite(FileName, screen)

            elif key.char == ('a'): # steer left
                FileName = "images/left/" + datestring
                screen = grab_screen(monitorNumber)
                cv2.imwrite(FileName, screen)

            elif key.char == ('d'): # steer left
                FileName = "images/right/" + datestring
                screen = grab_screen(monitorNumber)
                cv2.imwrite(FileName, screen)

            elif key.char == 'q': # quit
                return False
        if key == keyboard.Key.esc:
            return False

def on_release(key):
    if hasattr(key, 'char'):
        if key.char == 'q':
            # Stop listener
            return False
    if key == keyboard.Key.esc:
        return False

# Main function
if __name__ == "__main__":
    with mss.mss() as sct:
        print("Screens are recorded on pressing w,a,s,d. Press q for quit")
        time.sleep(5)
        with keyboard.Listener(on_press = on_press, on_release = on_release) as listener: listener.join()


