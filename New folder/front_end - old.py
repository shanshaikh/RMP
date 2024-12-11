from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import requests

root = Tk()
root.title('Project')

image_list = []
image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
url = ''

my_label = Label()
my_label.grid(row=0, column=0, columnspan=3)

def start(directory_path):
    global image_path
    global image_list
    global my_label

    image_path = directory_path
    files = os.listdir(image_path)
    image_files = [f for f in files if f.lower().endswith(image_extensions)]
    image_list.clear()
    for image_file in image_files:
        image_list.append(ImageTk.PhotoImage(Image.open(image_path + '/' + image_file)))
    my_label = Label(image=image_list[0])
    my_label.grid(row=0, column=0, columnspan=3)
    
def forward(image_number):
    global my_label
    global button_forward
    global button_back

    my_label.grid_forget()
    my_label = Label(image=image_list[image_number-1])
    button_forward = Button(root, text='>>', command=lambda: forward(image_number+1))
    button_back = Button(root, text='<<', command=lambda: back(image_number-1))

    if image_number == len(image_list):
            button_forward = Button(root, text='>>', state=DISABLED)
            
    my_label.grid(row=0, column=0, columnspan=3)
    button_back.grid(row=1, column=0)
    button_forward.grid(row=1, column=2)


def back(image_number):
    global my_label
    global button_forward
    global button_back

    my_label.grid_forget()
    my_label = Label(image=image_list[image_number-1])
    button_forward = Button(root, text='>>', command=lambda: forward(image_number+1))
    button_back = Button(root, text='<<', command=lambda: back(image_number-1))

    if image_number == 1:
            button_back = Button(root, text='<<', state=DISABLED)

    my_label.grid(row=0, column=0, columnspan=3)
    button_back.grid(row=1, column=0)
    button_forward.grid(row=1, column=2)

def get_Directory():
    image_path = filedialog.askdirectory(title='Select a Folder')
    start(image_path)

def get_Report():
    my_label
    response = requests.post(url, files=files)

button_back = Button(root, text="<<", command=back, state=DISABLED)
button_forward = Button(root, text=">>", command=lambda: forward(2))
button_selectDirectory = Button(root, text="Direc", command=get_Directory)
button_getReport = Button(root, text="Get Report", command=get_Report)

button_back.grid(row=1, column=0)
button_forward.grid(row=1, column=2)
button_selectDirectory.grid(row=2, column=0)
button_selectDirectory.grid(row=2, column=1)

start(os.getcwd())

root.mainloop()
