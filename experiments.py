# load and show an image with Pillow
from PIL import Image
import numpy as np

LOGO_IMAGE = "double-logo.png"

def image_to_bin(image: Image.Image):
    WHITE = [255,255,255,255]
    BLACK = [0,0,0,255]
    
    image_data = np.asarray(image)
    
    bin = []
    
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            color = image_data[x,y]
            if (color == WHITE).all():
                bin.append(1)
            elif (color == BLACK).all():
                bin.append(0)
            else:
                raise ValueError(f"Unknown color: {color}")
            
    return bin

def bin_to_image(bin: list):
    WHITE = [255,255,255,255]
    BLACK = [0,0,0,255]    
    
    image_data = np.zeros((1024, 512, 4), dtype=np.uint8)
    
    i = 0
    for x in range(image_data.shape[0]):
        for y in range(image_data.shape[1]):
            bit = bin[i]
            if bit == 1:
                image_data[x,y] = WHITE
            else:
                image_data[x,y] = BLACK
            
            i += 1
            
    return Image.fromarray(image_data)

# # Open the image form working directory
# image = Image.open(LOGO_IMAGE)

# image.show()

# bin = image_to_bin(image)

# image2 = bin_to_image(bin)

# image2.show()








# # create Pillow image
# image2 = Image.fromarray(data)
# print(type(image2))

# # summarize image details
# print(image2.mode)
# print(image2.size)