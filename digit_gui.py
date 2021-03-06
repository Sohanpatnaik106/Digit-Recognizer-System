# Import the required packages and libraries

import tensorflow as tf 
from tensorflow.keras.models import load_model
import numpy as np 
import os
import PIL
from PIL import ImageTk, Image, ImageDraw
import tkinter as tk 
from tkinter import *
from tensorflowTesting import testing
import cv2

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
width = 500
height = 500
center = height//2
white = (255, 255, 255)
green = (0, 128, 0)

def paint(event):
	x1, y1 = (event.x - 10), (event.y - 10)
	x2, y2 = (event.x + 10), (event.y + 10)
	cv.create_oval(x1, y1, x2, y2, fill = "black", width = 10)
	draw.line([x1, y1, x2, y2], fill = "black", width = 10)

def model():
	filename = "image.png"
	image1.save(filename)
	pred = testing()

	txt.insert(tk.INSERT, "The predicted value : {}".format(classes[np.argmax(pred[0])]))

def clear():
	cv.delete("all")
	draw.rectangle((0, 0, 500, 500), fill = (255, 255, 255, 0))
	txt.delete('1.0', END)

root = tk.Tk()
root.resizable(0, 0)
cv = tk.Canvas(root, width = width, height = height, bg = "white")
cv.pack()

image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

txt = tk.Text(root, bd = 3, exportselection = 0, bg = "white", font = "Helvetica", padx = 10, pady = 10, height = 5, width = 20)

cv.pack(expand = YES, fill = BOTH)
cv.bind("<B1-Motion>", paint)

btnModel = tk.Button(text = "Predict", command = model)
btnClear = tk.Button(text = "Clear", command = clear)

btnModel.pack()
btnClear.pack()

txt.pack()
root.title("Digit Recognizer ------ Sohan Patnaik")
root.mainloop()




