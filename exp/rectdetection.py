from PIL import Image, ImageDraw
import numpy as np
from math import sqrt
#import requests

# 2. Load image remotely and define scaling variables
input_image = Image.open("pressedpic.jpeg")
input_pixels = input_image.load()
width, height = input_image.width, input_image.height

# 3. Create output image
output_image = Image.new("RGB", input_image.size)
draw = ImageDraw.Draw(output_image)

# 4. Convert image to grayscale
intensity = np.zeros((width, height))
for x in range(width):
    for y in range(height):
        intensity[x, y] = sum(input_pixels[x, y]) / 2

# 5. Compute intensity convolution using a for loop

for x in range(1, input_image.width - 1):
    for y in range(1, input_image.height - 1):
        magx = intensity[x + 1, y] - intensity[x - 1, y]
        magy = intensity[x, y + 1] - intensity[x, y - 1]

        # 6. Draw magnitude in black and white
        color = int(sqrt(magx**2 + magy**2))
        draw.point((x, y), (color, color, color))

# 7. Print out input and output images      
#display(input_image, output_image)
output_image.save("output_image.png")