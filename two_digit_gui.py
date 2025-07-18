import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

MODEL_INPUT_SIZE = 128

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Two-Digit Live Recognition Software")
        master.configure(bg="#f0f0f0")

        self.canvas = tk.Canvas(master, width=400, height=400, bg="white", cursor="cross")
        self.canvas.pack(pady=10)

        self.slider_width = tk.Scale(master, from_=1, to=20, orient='horizontal', label='Stroke Width')
        self.slider_width.set(8)
        self.slider_width.pack()

        self.prediction_label = tk.Label(master, text="Prediction: None", font=("Helvetica", 36, "bold"), bg="#f0f0f0", fg="#333")
        self.prediction_label.pack()

        # Frame for progress bar and percentage label side by side
        progress_frame = tk.Frame(master, bg="#f0f0f0")
        progress_frame.pack(pady=10)

        self.confidence_bar = Progressbar(progress_frame, length=300, mode='determinate')
        self.confidence_bar.pack(side=tk.LEFT)

        self.confidence_label = tk.Label(progress_frame, text="0%", font=("Helvetica", 14), bg="#f0f0f0")
        self.confidence_label.pack(side=tk.LEFT, padx=10)

        self.btn_clear = tk.Button(master, text="Clear", command=self.clear_canvas, width=12)
        self.btn_clear.pack(pady=10)

        self.image = Image.new("L", (400, 400), 255)
        self.draw_obj = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_pos)

        self.model = load_model("cnn_model_from_images.h5")

    def draw(self, event):
        x, y = event.x, event.y
        r = self.slider_width.get()

        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=r*2, fill='black', capstyle=tk.ROUND, smooth=True)
            self.draw_obj.line([self.last_x, self.last_y, x, y], fill=0, width=r*2)
        else:
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
            self.draw_obj.ellipse([x - r, y - r, x + r, y + r], fill=0)

        self.last_x, self.last_y = x, y

        self.live_predict()

    def reset_last_pos(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        if messagebox.askyesno("Clear Canvas", "Clear the canvas?"):
            self.canvas.delete("all")
            self.image = Image.new("L", (400, 400), 255)
            self.draw_obj = ImageDraw.Draw(self.image)
            self.prediction_label.config(text="Prediction: None")
            self.confidence_bar['value'] = 0
            self.confidence_label.config(text="0%")

    def live_predict(self):
        img = self.image.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        img = img.reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1)

        prediction = self.model.predict(img)[0]
        digit = np.argmax(prediction)
        confidence = prediction[digit] * 100

        self.prediction_label.config(text=f"Prediction: {digit:02d}")
        self.confidence_bar['value'] = confidence
        self.confidence_label.config(text=f"{confidence:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
