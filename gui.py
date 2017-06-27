import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image, ImageFilter
import numpy as np
from skimage import feature
from ftdetect import features
from scipy import ndimage
from skimage.filters import roberts, sobel, scharr, prewitt, threshold_otsu

import part1 as p1
import part2 as p2

import time

def open_file():
    filename = filedialog.askopenfilename()
    img = Image.open(filename)

    scaled = img.resize((250, 250), Image.ANTIALIAS)
    local_img = ImageTk.PhotoImage(scaled)
    display_img.config(image=local_img)
    display_img.image = local_img
    original_label.config(text="Original Image")

    threshold_slider.grid(row=0, column=1, sticky='NE')
    threshold_slider.config(label='Threshold', length=255)
    threshold_slider.set(15)

    popovici_button.grid(row=0, column=2, sticky='NW')
    ca_button.grid(row=0, column=2, sticky='SW')

    denoise_mode_button.grid(row=0, column=3, sticky='NW')
    denoise_gauss_button.grid(row=0, column=3, sticky='SW')
    denoise_median_button.grid(row=0, column=3, sticky='NE')

    display_denoised.config(image=None)
    display_denoised.image=None

    part2_display.config(image=None)
    part2_display.image=None

    global img_grey
    global denoised
    img_grey = np.asarray(img.convert('L'))
    img_grey.setflags(write=1)

    global initial_otsu
    initial_otsu = threshold_otsu(img_grey)

    denoised = np.copy(img_grey)

    popovici(img_grey)
    canny_convert(img_grey)
    #susan_convert(img_grey)
    sobel_convert(img_grey)
    roberts_convert(img_grey)

def canny_convert(input):
    edges = feature.canny(input)

    edges = (edges*255).astype('uint8');

    display = Image.fromarray(edges)
    scaled = display.resize((250, 250), Image.ANTIALIAS)
    local_img = ImageTk.PhotoImage(scaled)

    display_canny.config(image=local_img)
    display_canny.image = local_img
    canny_label.config(text='Canny Edge Detection')

def susan_convert(input):
    edges = features.susanEdge(input)

    display = Image.fromarray(edges*255)
    scaled = display.resize((250, 250), Image.ANTIALIAS)
    local_img = ImageTk.PhotoImage(scaled)

    display_susan.config(image=local_img)
    display_susan.image = local_img


def sobel_convert(input):
    edges = sobel(input)
    edges = (edges*255).astype('uint8')

    display = Image.fromarray(edges)
    scaled = display.resize((250, 250), Image.ANTIALIAS)
    local_img = ImageTk.PhotoImage(scaled)

    sobel_label.config(text='Sobel Edge Detection')

    display_sobel.config(image=local_img)
    display_sobel.image = local_img


def popovici(input):
    edges = p1.ca_edge(input, epsilon.get())

    edges = edges.astype('uint8')

    display = Image.fromarray(edges)
    scaled = display.resize((250, 250), Image.ANTIALIAS)
    local_img = ImageTk.PhotoImage(scaled)
    ca_label.config(text='Part 1 CA Edge Detection (Popovici)')

    display_ca.config(image=local_img)
    display_ca.image = local_img


def ca_denoise(input, type):	 

    global denoised
    if type == 'gauss':   
    	denoised = p2.denoise_gaussian(input)*255
    elif type == 'mode':
    	denoised = p1.denoise_mode(input)
    elif type == 'median':
        denoised = p2.denoise_median(input)

    display = Image.fromarray(denoised.astype('uint8'))

    scaled = display.resize((250, 250), Image.ANTIALIAS)
    local_img = ImageTk.PhotoImage(scaled)

    display_denoised.config(image=local_img)
    display_denoised.image = local_img
    denoised_label.config(text='Denoised image')


def ca_convert(input):
    marked = p2.mark_neighbourhoods(input, initial_otsu)

    display = Image.fromarray(marked.astype('uint8'))

    scaled = display.resize((250, 250), Image.ANTIALIAS)
    local_img = ImageTk.PhotoImage(scaled)

    part2_display.config(image=local_img)
    part2_display.image = local_img
    part2_label.config(text='Part 2 CA Edge Detection')

def roberts_convert(input):
    edges = roberts(input)
    edges = (edges*255).astype('uint8')

    display = Image.fromarray(edges)
    scaled = display.resize((250, 250), Image.ANTIALIAS)
    local_img = ImageTk.PhotoImage(scaled)

    roberts_label.config(text='Roberts Cross Edge Detection')

    display_roberts.config(image=local_img)
    display_roberts.image = local_img

root = tk.Tk();
root.title('Edge Detection using Cellular Automata')
root.geometry('1050x650+150+35')

tk.Button(root, text='Open File...', command=open_file).grid(row=0, column=0, sticky='NW')

epsilon = tk.IntVar()
threshold_slider = tk.Scale(root, from_=0, to=255, orient=tk.HORIZONTAL, variable=epsilon)
popovici_button = tk.Button(root, text='Part 1 Edge Detection', command=lambda: popovici(denoised))
ca_button = tk.Button(root, text='Part 2 Edge Detection', command=lambda: ca_convert(denoised))

denoise_mode_button = tk.Button(root, text='Mode Filter', command=lambda: ca_denoise(denoised, 'mode'))
denoise_gauss_button = tk.Button(root, text='Gaussian Filter', command=lambda: ca_denoise(denoised, 'gauss'))
denoise_median_button = tk.Button(root, text='Median Filter', command=lambda: ca_denoise(denoised, 'median'))

display_img = tk.Label(root)
display_img.grid(row=1, column=0)

display_canny = tk.Label(root)
display_canny.grid(row=4, column=0)

display_susan = tk.Label(root)
display_susan.grid(row=4, column=1)
susan_label = tk.Label(root)
susan_label.grid(row=5,column=1)

display_ca = tk.Label(root)
display_ca.grid(row=1, column=1)
ca_label = tk.Label(root)
ca_label.grid(row=2, column=1)

display_denoised = tk.Label(root)
display_denoised.grid(row=1, column=2)
denoised_label = tk.Label(root)
denoised_label.grid(row=2, column=2)

display_sobel = tk.Label(root)
display_sobel.grid(row=4, column=3)
sobel_label = tk.Label(root)
sobel_label.grid(row=5,column=3)

display_roberts = tk.Label(root)
display_roberts.grid(row=4, column=2)
roberts_label = tk.Label(root)
roberts_label.grid(row=5,column=2)

original_label = tk.Label(root)
original_label.grid(row=2, column=0)

canny_label = tk.Label(root)
canny_label.grid(row=5, column=0)

ca_time = tk.Label(root)
ca_time.grid(row=3, column=1)

canny_time = tk.Label(root)
canny_time.grid(row=3, column=2)

part2_display = tk.Label(root)
part2_display.grid(row=1, column=3)
part2_label = tk.Label(root)
part2_label.grid(row=2, column=3)

root.mainloop();
