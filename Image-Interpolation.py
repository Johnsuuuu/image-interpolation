import cv2
import numpy as np
import time
import math


def Nearest_Neighbor_Interpolation(img,height,width,channels):
    new_img = np.zeros((height,width,channels), dtype = np.uint8)

    for i in range(0,height):
        for j in range(0,width):
            y = i*(img.shape[0]/height)
            x = j*(img.shape[1]/width)
            new_x = round(x)
            new_y = round(y)
            if new_y == img.shape[0] or new_x == img.shape[1]:
                new_x = new_x -1
                new_y = new_y -1
            new_img[i][j] = img[new_y][new_x]

    return new_img


def Bilinear_Interpolation(img,height,width,channels):
    new_img = np.zeros((height,width,channels), dtype=np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            y = i * (img.shape[0]/height)
            x = j * (img.shape[1] / width)
            y_int = int(y)
            x_int = int(x)
            u = y - y_int
            v = x - x_int
            if y_int == img.shape[0] - 1 or x_int == img.shape[1] - 1:
                y_int = y_int - 1
                x_int = x_int - 1

            new_img[i][j] = (x_int+1-x)*(y_int+1-y)*img[y_int][x_int] + (x-x_int)*(y_int+1-y)*img[y_int][x_int+1] + (x_int+1-x)*(y-y_int)*img[y_int+1][x_int]+(y-y_int)*(x-x_int)*img[y_int+1][x_int+1]

    return new_img


#convolution kernel
def S(x):
    x = np.abs(x)

    if 0 <= x <= 1:
        return 1-2.5*x*x+1.5*x*x*x
    elif 1 < x <= 2:
        return 2-4*x+2.5*x*x-0.5*x*x*x
    else:
        return 0


def Bicubic_Interpolation(img,height,width,channels):
    new_img = np.zeros((height, width, channels), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            y = i * (img.shape[0] / height)
            x = j * (img.shape[1] / width)
            y_int = int(y)
            x_int = int(x)
            u = y - y_int
            v = x - x_int
            tmp=0

            for m in range(-1,3):
                for n in range(-1,3):
                    if y_int+m<0 or x_int+n<0 or y_int+m>=img.shape[0] or x_int+n>=img.shape[1]:
                        continue
                    tmp+=img[y_int+m][x_int+n]*S(m-u)*S(n-v)
            new_img[i][j]=np.clip(tmp,0,255)

    return new_img




def psnr(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse == 0:
        return "same image"
    PIXEL_MAX = 255.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == '__main__':
    img = cv2.imread("einstein.jpg", cv2.IMREAD_COLOR)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = img.shape

    new_dimension = (500,500)
    orig = cv2.resize(img, new_dimension, interpolation=cv2.INTER_LANCZOS4)
    resize_height = 500
    resize_width = 500

    start = time.time()
    new_image1 = Nearest_Neighbor_Interpolation(img,resize_height,resize_width,channels)
    end = time.time()
    time1 = end - start

    start = time.time()
    new_image2 = Bilinear_Interpolation(img,resize_height,resize_width,channels)
    end = time.time()
    time2 = end - start

    start = time.time()
    new_image3 = Bicubic_Interpolation(img, resize_height, resize_width,channels)
    end = time.time()
    time3 = end - start

    psnr1 = psnr(new_image1, orig)
    psnr2 = psnr(new_image2, orig)
    psnr3 = psnr(new_image3, orig)

    print(psnr1, psnr2, psnr3)
    print(time1, time2, time3)

    cv2.imshow("img", img)
    cv2.imshow("new image1", new_image1)
    cv2.imshow("new image2", new_image2)
    cv2.imshow("new image3", new_image3)

    cv2.imwrite("einstein_1.jpg", new_image1)
    cv2.imwrite("einstein_2.jpg", new_image2)
    cv2.imwrite("einstein_3.jpg", new_image3)

    cv2.waitKey(0)