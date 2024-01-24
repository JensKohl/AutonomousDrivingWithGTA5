"""
This module grab screens in GTA Vice City when the vehicle moves..
"""

import time
import os
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

# since i had several driving accidents, a function to delete the last x pics is useful
IMAGE_RING_BUFFER_SIZE = 10000 # how many images to remember
#current_image_counter = 0
image_ring_buffer = [] # store name of saved images, so we can easily delete them after crash
# NOTE: could be done without ringbuffer, but then we spend little bit more memory than needed...

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

def erase_last_pics(amount_images):
    """
    deletes the last images in case of an accident happening during driving
    Args:
        amount_images(int): amount of images to delete
    Returns:
        screen: image
    """
    # we could do this without wrap_around case because since we are
    # driving a long time, a few images unnecessarily lost don't really mind
    if (current_image_counter - 1 - amount_images) < 0:
        print("WRAPAROUND")
        wrap_around = IMAGE_RING_BUFFER_SIZE - current_image_counter

        for i in range(current_image_counter - 1, 0, -1):
            print(image_ring_buffer[i-1])
            try:
                os.remove(image_ring_buffer[i])
            except OSError:
                pass

        for j in range(IMAGE_RING_BUFFER_SIZE - 1, IMAGE_RING_BUFFER_SIZE - 1 - wrap_around, -1):
            print(image_ring_buffer[j-1])
            try:
                os.remove(image_ring_buffer[j-1])
            except OSError:
                pass

    else:   # no wraparound, delete images
        print(current_image_counter)
        new_index = current_image_counter - 1 - amount_images
        for i in range(current_image_counter-1, new_index, -1):
            print(image_ring_buffer[i-1])
            try:
                os.remove(image_ring_buffer[i-1])
            except OSError:
                pass
        return new_index


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

            # TODO: here is the ringbuffer counter
            global current_image_counter

            if key.char == ("w"):  # accelerate
                file_name = "images/accelerate/" + datestring
                screen = grab_screen(monitor_number)
                cv2.imwrite(file_name, screen)
                image_ring_buffer.append(file_name)
                current_image_counter = (current_image_counter + 1) % IMAGE_RING_BUFFER_SIZE
                return True

            # braking currently disabled
            #if key.char == ("s"):  # brake
            #    file_name = "images/decelerate/" + datestring
            #    screen = grab_screen(monitor_number)
            #    cv2.imwrite(file_name, screen)
            #    image_ring_buffer.append(file_name)
            #   current_image_counter = (current_image_counter + 1) % IMAGE_RING_BUFFER_SIZE
            #    return True

            if key.char == ("a"):  # steer left
                file_name = "images/left/" + datestring
                screen = grab_screen(monitor_number)
                cv2.imwrite(file_name, screen)
                image_ring_buffer.append(file_name)
                current_image_counter = (current_image_counter + 1) % IMAGE_RING_BUFFER_SIZE
                return True

            if key.char == ("d"):  # steer left
                file_name = "images/right/" + datestring
                screen = grab_screen(monitor_number)
                cv2.imwrite(file_name, screen)
                image_ring_buffer.append(file_name)
                current_image_counter = (current_image_counter + 1) % IMAGE_RING_BUFFER_SIZE
                return True

            # allows to delete the last pics in case of an accident
            if key.char == "k":
                current_image_counter -= 1
                return_value = erase_last_pics(5)
                current_image_counter = return_value
                return True

            if key.char == "x":  # quit
                current_image_counter -= 1
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
        if key.char == "x": # switch from q to x...q changes radio stations in GTA :-)
            # Stop listener
            return False
    if key == keyboard.Key.esc:
        return False
    return True


# Main function
if __name__ == "__main__":
    global current_image_counter
    current_image_counter = 0

    with mss.mss() as sct:
        print("Record screens on pressing w,a,s,d. Press k to erase last 5 pics. Press x for quit")
        time.sleep(5)
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
