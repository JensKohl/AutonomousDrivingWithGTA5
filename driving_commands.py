"""
Module providing the relevant keys to steer/ drive the vehicle into a direction.
Grabs an image, feeds image to CNN-model,
finds out direction for driving,
and steers vehicle via key press.
"""

import time
from directkeys import PressKey, ReleaseKey, W, A, S, D

def accelerate():
    """function to give command accelerate via keypress
    Args:
        -
    Returns:
        -
    """
    print("accelerate")
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def decelerate():
    """function to give command decelerate via keypress
    Args:
        -
    Returns:
        -
    """
    print("decelerate")
    PressKey(S)
    # ReleaseKey(W)
    # ReleaseKey(A)
    # ReleaseKey(D)
    ReleaseKey(S)


def steer_left():
    """function to give command steer left via keypress
    Args:
        -
    Returns:
        -
    """
    print("driving left")
    PressKey(W)
    PressKey(A)
    time.sleep(0.1)
    ReleaseKey(W)
    ReleaseKey(A)
    #ReleaseKey(D)
    # ReleaseKey(S)


def steer_right():
    """function to give command steer right via keypress
    Args:
        -
    Returns:
        -
    """
    print("driving right")
    PressKey(W)
    PressKey(D)
    time.sleep(0.1)
    ReleaseKey(W)
    #ReleaseKey(A)
    ReleaseKey(D)
    # ReleaseKey()
