import tkinter as tk
import subprocess
import t1

root = tk.Tk()
root.title("我的窗口")
root.geometry("300x200")


def button_click():
    print("按钮被点击了！")
    root.destroy()
    # t1.f()
    subprocess.call(['python', 't1.py'])


button = tk.Button(root, text="点击我", command=button_click)
button.place(x=100, y=160)
# button.grid(row=10, column=20)
# button.pack()
root.mainloop()
