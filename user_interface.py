from tkinter import *
from PIL import ImageTk, Image
from datetime import datetime
import time


root = Tk()
root.title("Recognize Face System")
root.geometry('300x500')


def register():
    pass


def check():
    pass


logo = ImageTk.PhotoImage(Image.open('logo.jpg'))
label1 = Label(root, image=logo, height=300, width=300).grid(row=0, column=0)
label2 = Label(root).grid(row=0, column=1)
register_button = Button(root, text="Register", command="register")
check_button = Button(root, text="Check", command="check")
register_button.grid(row=1, column=0, pady=30)
check_button.grid(row=2, column=0, pady=30)


root.mainloop()
