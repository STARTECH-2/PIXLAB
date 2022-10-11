import cv2
import imutils
import easygui
import imageio
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import pandas as pd
from skimage.restoration import inpaint
from skimage.transform import resize
from skimage import color
from skimage.restoration import denoise_tv_chambolle

def uploadToCartoonify():
    ImagePath=easygui.fileopenbox()
    cartoonify(ImagePath)

# def uploadToColorize():
#     ImagePath=easygui.fileopenbox()
#     Colorize(ImagePath)

def Colorize(ImagePath):
    net = cv2.dnn.readNetFromCaffe('G:/Projects/PIXLAB/Colorize/colorization_deploy_v2.prototxt', 'G:/Projects/PIXLAB/Colorize/colorization_release_v2.caffemodel')
    pts = np.load('G:/Projects/PIXLAB/Colorize/pts_in_hull.npy')
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]
    image = cv2.imread(ImagePath)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    cv2.imshow("Original", image)
    cv2.imshow("Colorized", colorized)
    cv2.waitKey(0)


def cartoonify(ImagePath):
    originalimage = cv2.imread(ImagePath)
    originalimage = cv2.cvtColor(originalimage, cv2.COLOR_BGR2RGB)
    if originalimage is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()
    ReSized1 = cv2.resize(originalimage, (940, 610))
    # converting an image to grayscale
    grayScaleImage = cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)
    ReSized2 = cv2.resize(grayScaleImage, (940, 610))
    # applying median blur to smoothen an image
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    ReSized3 = cv2.resize(smoothGrayScale, (940, 610))
    # retrieving the edges
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 9)
    ReSized4 = cv2.resize(getEdge, (940, 610))
    # applying bilateral filter to remove noise and keep edge sharp as required
    colorImage = cv2.bilateralFilter(originalimage, 9, 300, 300)
    ReSized5 = cv2.resize(colorImage, (940, 610))
    # masking edged image with our "BEAUTIFY" image
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
    ReSized6 = cv2.resize(cartoonImage, (940, 610))
    images = [ReSized1, ReSized2, ReSized3, ReSized4, ReSized5, ReSized6]
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
    save1 = Button(top, text="Save cartoon image", command=lambda: save(ReSized6, ImagePath), padx=30, pady=5)
    save1.configure(background='#364156', foreground='white', font=('calibri', 10, 'bold'))
    save1.pack(side=TOP, pady=50)
    plt.show()


def save(ReSized6, ImagePath):
    newName = "cartoon_Image"
    path1 = os.path.dirname(ImagePath)
    extension = os.path.splitext(ImagePath)[1]
    path = os.path.join(path1, newName + extension)
    cv2.imwrite(path, cv2.cvtColor(ReSized6, cv2.COLOR_RGB2BGR))
    I = "Image saved by name " + newName + " at " + path
    tk.messagebox.showinfo(title=None, message=I)


def show_image(image, title='Image', cmap_type='gray'):
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')


def plot_comparison(img_original, img_filtered, img_title_filtered):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 8), sharex=True, sharey=True)
    ax1.imshow(img_original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(img_filtered, cmap=plt.cm.gray)
    ax2.set_title(img_title_filtered)
    ax2.axis('off')

print("\n----------------------------------")
print("PIXLAB - An Image Processing Forum")
print("----------------------------------\n")
print("1. Restore The Image")
print("2. Colorize The Image")
print("3. Cartoonify the Image")
n = int(input())
if n == 1:
    plt.rcParams['figure.figsize'] = (10, 8)
    print('')
    print("1. Enhance The Image")
    print("2. Remove Logos")
    print("3. Remove Noise")
    print('')
    t = int(input())
    if t == 1:
        # Enhancing image
        defect_image = plt.imread('./dataset/damaged_astronaut.png')
        defect_image = resize(defect_image, (512, 512))
        defect_image = color.rgba2rgb(defect_image)
        mask = pd.read_csv('./dataset/astronaut_mask.csv').to_numpy()
        restored_image = inpaint.inpaint_biharmonic(defect_image, mask, multichannel=True)
        plot_comparison(defect_image, restored_image, 'Restored image')

    elif t == 2:
        # Removing logos
        image_with_logo = plt.imread('G:/Projects/PIXLAB/Image Restoration/Logo/TEST-2.jpg')
        mask = np.zeros(image_with_logo.shape[:-1])
        mask[210:272, 360:425] = 1
        image_logo_removed = inpaint.inpaint_biharmonic(image_with_logo, mask, multichannel=True)
        plot_comparison(image_with_logo, image_logo_removed, 'Image with logo removed')

    elif t == 3:
        # Reducing Noise
        noisy_image = plt.imread('./dataset/miny.jpeg')
        denoised_image = denoise_tv_chambolle(noisy_image, multichannel=True)
        plot_comparison(noisy_image, denoised_image, 'Denoised Image')

    else:
        print("ERROR TRY AGAIN")

elif n==2:
    Colorize("G:/Projects/PIXLAB/Colorize/TEST-3.jpg")

elif n==3:
    top = tk.Tk()
    top.geometry('400x400')
    top.title('Cartoonify your Image !')
    top.configure(background='white')
    label = Label(top, background='#CDCDCD', font=('calibri', 20, 'bold'))
    upload = Button(top, text="Cartoonify an Image", command=uploadToCartoonify, padx=10, pady=5)
    upload.configure(background='#364156', foreground='white', font=('calibri', 10, 'bold'))
    upload.pack(side=TOP, pady=50)
    top.mainloop()
else:
    print("ERROR TRY AGAIN")

