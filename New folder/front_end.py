from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import requests

root = Tk()
root.title('Project')

image_list = []
image_files = []
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
url = 'http://127.0.0.1:8090/upload'
image_pos = 0

my_label = Label()
my_label.grid(row=0, column=0, columnspan=3)

def start(directory_path):
    global image_path
    global image_list
    global my_label
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

def traverse(image_number):
    global my_label
    global button_forward
    global button_back
    global image_pos
    global button_back
    global button_forward
    
    image_pos += image_number
    my_label.grid_forget()
    my_label = Label(image=image_list[image_pos])
    button_forward['state'] = NORMAL
    button_back['state'] = NORMAL

    if image_pos == (len(image_list)-1):
        button_forward['state'] = DISABLED
            
    if image_pos == 0:
        button_back['state'] = DISABLED
            
    my_label.grid(row=0, column=0, columnspan=3)

def get_Directory():
    image_path = filedialog.askdirectory(title='Select a Folder')
    start(image_path)

def get_Report():
    global image_path
    global image_list
    global image_pos
    global image_files

    print(image_files)
    imagePathToSend = image_path + '/' + image_files[image_pos]
    print(imagePathToSend)
##    files = {"image": open(imagePathToSend, "rb")}
##    response = requests.post(url, files=files)
    print(url)
    with open(imagePathToSend, "rb") as f:
        response = requests.post(url, data=f.read())
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

button_back = Button(root, text="<<", command=lambda: traverse(-1), state=DISABLED)
button_forward = Button(root, text=">>", command=lambda: traverse(1))
button_selectDirectory = Button(root, text="Direc", command=get_Directory)
button_getReport = Button(root, text="Get Report", command=get_Report)

button_back.grid(row=1, column=0)
button_forward.grid(row=1, column=2)
button_selectDirectory.grid(row=2, column=0)
button_getReport.grid(row=2, column=1)

start(os.getcwd())

root.mainloop()
