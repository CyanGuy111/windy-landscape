import tkinter as tk
from main import *

class DrawCanvas(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("This works somehow")

        self.pixel_size = 10
        self.grid_size = 28

        canvas_size = self.pixel_size * self.grid_size

        self.canvas = tk.Canvas(self, width=canvas_size, height=canvas_size, bg="white")
        self.canvas.pack()

        self.pixels = [[False]*self.grid_size for _ in range(self.grid_size)]

        for i in range(self.grid_size + 1):
            x = i * self.pixel_size
            self.canvas.create_line(x, 0, x, canvas_size, fill="lightgray")
            self.canvas.create_line(0, x, canvas_size, x, fill="lightgray")

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)

        button = tk.Button(self, text = "Confirm", command = lambda: self.forw(rinp=self.pixels))
        button.pack()

        button2 = tk.Button(self, text = "Reset", command = self.reset_canvas)
        button2.pack()

        self.output = tk.Text(self, height=1)
        self.output.pack()

    def forw(self, rinp):
        inp = np.array(rinp, dtype=int)
        inp = inp.flatten()
        pred = run(inp)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, pred)

    def draw(self, event):
        col = event.x // self.pixel_size
        row = event.y // self.pixel_size

        brush_size = 3
        offset = brush_size // 2

        for dy in range(-offset, offset + 1):
            for dx in range(-offset, offset + 1):
                r = row + dy
                c = col + dx

                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    if not self.pixels[r][c]:
                        x1 = c * self.pixel_size
                        y1 = r * self.pixel_size
                        x2 = x1 + self.pixel_size
                        y2 = y1 + self.pixel_size

                        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="")
                        self.pixels[r][c] = True
            
    def reset_canvas(self):
        self.canvas.delete("all")
        self.pixels = [[False] * self.grid_size for _ in range(self.grid_size)]

        for i in range(self.grid_size + 1):
            x = i * self.pixel_size
            self.canvas.create_line(x, 0, x, self.pixel_size * self.grid_size, fill="lightgray")
            self.canvas.create_line(0, x, self.pixel_size * self.grid_size, x, fill="lightgray")

if __name__ == "__main__":
    app = DrawCanvas()
    app.mainloop()
