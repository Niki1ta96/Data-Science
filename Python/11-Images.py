from skimage import data
from skimage import io
from skimage import color
import os 
import pandas as pd

coffee = data.coffee()
type(coffee)
coffee.shape
#(400L, 600L, 3L) --- # 3L is dimensions, 3D array
io.imshow(coffee)

coffee_grey = color.rgb2gray(coffee)
type(coffee_grey)
coffee_grey.shape     #(400L, 600L) --- # 2D array
io.imshow(coffee_grey)

tmp1= coffee_grey.reshape((1, -1))
type(tmp1)
tmp1.shape         #(1L, 240000L)

tmp2 = tmp1.reshape((400, 600))
tmp2.shape          #(400L, 600L)
io.imshow(tmp2)

os.getcwd()
os.chdir("C:\\Users\\nikita.jaikar\\Documents\\GitHub\\Data-Science\\Python")

digit_train = pd.read_csv("C:\\Users\\nikita.jaikar\\Documents\\GitHub\\Data-Science\\Data\\digit_train.csv")
digit_train.shape

image = digit_train.iloc[0,1:]
image.shape
image_original = image.reshape([28,28])/255.0
io.imshow(image_original)






