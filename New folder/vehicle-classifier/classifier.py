from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import sys
import torch
from torchvision import models, transforms
import torch.nn as nn
from io import BytesIO
import requests

# Initialize Tkinter window
root = Tk()
root.title('Vehicle Classifier')

image_list = []
image_files = []
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
image_pos = 0

my_label = Label()
my_label.grid(row=0, column=0, columnspan=3)


# Start function (loads image files)
def start(directory_path):
    global image_path
    global image_list
    global image_pos
    global image_files
    
    image_pos = 0
    image_path = directory_path
    files = os.listdir(image_path)
    image_files = [f for f in files if f.lower().endswith(image_extensions)]
    image_list.clear()
    for image_file in image_files:
        image_list.append(ImageTk.PhotoImage(Image.open(image_path + '/' + image_file)))
    traverse(0)

# Traverse through images
def traverse(image_number):
    global my_label
    global button_forward
    global button_back
    global image_pos
    global button_back
    global button_forward

    image_pos += image_number
    my_label.grid_forget()
    if image_pos >= 0 and image_pos < len(image_list):
        my_label = Label(image=image_list[image_pos])
    button_forward['state'] = NORMAL
    button_back['state'] = NORMAL

    if image_pos >= (len(image_list)-1):
        button_forward['state'] = DISABLED
            
    if image_pos <= 0:
        button_back['state'] = DISABLED
            
    my_label.grid(row=0, column=0, columnspan=3)

# Select the directory
def get_Directory():
    image_path = filedialog.askdirectory(title='Select a Folder')
    start(image_path)

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Get the report by predicting using the model
def get_Report():
    global image_path
    global image_list
    global image_pos
    global image_files

    if not image_files or image_pos < 0 or image_pos >= len(image_files):
        result_label.config(text="No image selected for prediction.")
        return

    # Path of the selected image
    imagePathToSend = image_path + '/' + image_files[image_pos]

    # Backend API URL (Replace with your backend's actual IP or domain)
    backend_url = "http://18.212.33.234:5000/predict"

    try:
        # Send the image to the backend
        with open(imagePathToSend, 'rb') as image_file:
            response = requests.post(backend_url, files={"image": image_file})

        # Handle the backend's response
        if response.status_code == 200:
            data = response.json()
            if "prediction" in data:
                result_label.config(text=f"Prediction: {data['prediction']}")
            else:
                result_label.config(text="Error: No prediction received.")
        else:
            result_label.config(text=f"Error: {response.status_code} {response.text}")

    except Exception as e:
        result_label.config(text="Error communicating with the backend.")
        print(f"Error: {e}")


# UI elements
button_back = Button(root, text="<<", command=lambda: traverse(-1), state=DISABLED)
button_forward = Button(root, text=">>", command=lambda: traverse(1))
button_selectDirectory = Button(root, text="Select Directory", command=get_Directory)
button_getReport = Button(root, text="Get Report", command=get_Report)

button_back.grid(row=1, column=0)
button_forward.grid(row=1, column=2)
button_selectDirectory.grid(row=2, column=0)
button_getReport.grid(row=2, column=1)

result_label = Label(root, text="Prediction: ", font=("Helvetica", 14))
result_label.grid(row=3, columnspan=3)

# Start the app with the current directory
start(os.getcwd())

# Main loop
root.mainloop()
