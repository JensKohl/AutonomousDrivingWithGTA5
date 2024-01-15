# Lecture Data Management and technical applications

## Lesson 9 - CNN

This project shall show students how to use Convolutional Neural Networks for the very well known use case *Autonomous driving* on the example of GTA5.
Although -of course- GTA Vice City is still the best GTA ever, I had the well-known problems with the steam edition of GTA Vice City and a current Windows version.

That's why we are using GTA 5....

This work is under construction and in contrast to the lecture I am using here PyTorch.

How is it structured:
* directkeys.py: file with the key codes necessary to overload i.e. enable drive_vehicle.py to send key commands to GTA 5.
* drive_vehicle_CNN.py: this function needs to be run in background while GTA 5 is running. It grabs a screen from the GTA5 game and then analyses in which direction the car needs to drive by sending the grabbed screen as image to the model trained via Convolutional Neural Networkk (CNN). The model's output is then sent as driving command to the game.
* drive_vehicle_transfer_learning.py: does exactly the same, but uses the transfer learning model.
* driving_commands.py: function containing the right key press combinations for driving into a direction or accelerate or decelerate.
* get_training_data.py: gets the training data. This function needs to be run in background while GTA 5 is running. Whenever user uses a command to steer a vehicle, a screen shot is made and store in a specific directory. These images are used to train the model.
* model.py: the class file for the CNN model
* prepare_data.py: copoies the grabbed screens into a specific tree structure which can be used by data augmentations.
* test_model.py: functions to test the model with images 
* train_model_CNN.py: trains the CNN model
* train_transfer_learning.py: trains the Transfer learning model

## What needs yet to be done:
* improve model accuracy by getting more and more images and curating them: it's currently difficult for the model to distinguish between braking and accelerating...
* Train loooonger :-)
