# 诊断用动画源：一个持续运动的窗口（DDA 每帧都有内容更新，模拟用户看视频/游戏的场景）
import tkinter as tk


class Anim:
    def __init__(self):
        self.r = tk.Tk()
        self.r.title("diag-anim")
        self.r.geometry("800x600+100+100")
        self.c = tk.Canvas(self.r, bg="#202020", highlightthickness=0)
        self.c.pack(fill=tk.BOTH, expand=True)
        self.x, self.y, self.dx, self.dy = 400, 300, 7, 5
        self.ball = self.c.create_oval(0, 0, 40, 40, fill="#ff6030", outline="#ffffff")
        self.n = 0
        self.tick()

    def tick(self):
        w = self.c.winfo_width() or 800
        h = self.c.winfo_height() or 600
        self.x += self.dx
        self.y += self.dy
        if not 20 < self.x < w - 20:
            self.dx = -self.dx
        if not 20 < self.y < h - 20:
            self.dy = -self.dy
        self.c.coords(self.ball, self.x - 20, self.y - 20, self.x + 20, self.y + 20)
        self.n += 1
        if self.n % 60 == 0:
            self.c.itemconfigure(self.ball, fill="#30a0ff" if self.n % 120 == 0 else "#ff6030")
        self.r.after(8, self.tick)   # ~120Hz 驱动，DDA 稳定出 60Hz 帧


_a = Anim()
_a.r.mainloop()
