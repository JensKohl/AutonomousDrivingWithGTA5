import splitfolders

input_folder = 'images/'
splitfolders.ratio(input_folder, output="inputimages", seed=1337, ratio=(.6, .2, .2), group_prefix=None)