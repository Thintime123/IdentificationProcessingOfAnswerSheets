import tkinter as tk
import subprocess
import main

def button1_click():
    print('Button1 is clicked')
    exit(0)


def button2_click():
    print('Button2 is clicked and This program ends.')
    exit(0)


def finish():
    win = tk.Tk()
    win.title('End')
    win.geometry('400x150')

    def close_win():
        print('Right button is clicked.')
        win.destroy()

    text = '\n\tAutomatic identification and result feedback have been completed!'

    label = tk.Label(win, text=text, font=('Arial', 12, 'italic'), wraplength=400)
    label.config(justify='left')

    # button1 = tk.Button(win, text='exit', command=button1_click)
    # button1.place(x=130, y=130)

    button2 = tk.Button(win, text='thanks', command=close_win)
    button2.place(x=230, y=100)

    label.pack()
    win.mainloop()


def end():
    # 创建主窗口
    root = tk.Tk()
    root.title(" ")
    root.geometry("460x200")  # 设置窗口大小，格式为"宽度x高度"

    def loop():
        print('Once again\n')
        root.destroy()
        main.work()
        # subprocess.call(['python', 'main.py'])

    def close_win():
        print('Right button is clicked and this program ends.')
        root.destroy()

    content = (
        '  The automatic marking of papers has been completed this time. \n   Would you like to scan it again?')
    # 添加一个标签组件
    label = tk.Label(root, text=content, font=("Arial", 12, "italic"), wraplength=400)
    label.config(justify='left')

    # 按钮
    button1 = tk.Button(root, text='Yes', command=loop)
    button1.place(x=140, y=150)

    button2 = tk.Button(root, text='No', command=close_win)
    button2.place(x=260, y=150)

    label.pack()  # 将标签放置在窗口中

    # 进入主事件循环，使窗口保持显示状态
    root.mainloop()


if __name__ == "__main__":
    finish()
