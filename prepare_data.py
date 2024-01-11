"""
This module copies images into relevant folder structure
for image augmentation used when training.
"""

import splitfolders

INPUT_FOLDER = "images/"
splitfolders.ratio(
    INPUT_FOLDER,
    output="inputimages",
    seed=1337,
    ratio=(0.6, 0.2, 0.2),
    group_prefix=None,
)
