from tkinter import *
from PIL import ImageTk, Image
from tkinter import simpledialog, messagebox
from register import *


root = Tk()

root.title("Recognize Face System")
root.geometry('300x500')


def register():
    name = simpledialog.askstring(title="Entry name", prompt="What's your name")
    save_data(name)
    messagebox.showinfo("Information", "Register successfully")


def check():
    try:
        predict()
        if 0xFF == ord('q'):
            root.quit()
    except ValueError:
        messagebox.showinfo("Information", "Not found user")


logo = ImageTk.PhotoImage(Image.open('logo.jpg'))
label1 = Label(root, image=logo, height=300, width=300).grid(row=0, column=0)
label2 = Label(root).grid(row=0, column=1)
register_button = Button(root, text="Register", command=register)
check_button = Button(root, text="Check", command=check)
register_button.grid(row=1, column=0, pady=30)
check_button.grid(row=2, column=0, pady=30)


root.mainloop()
