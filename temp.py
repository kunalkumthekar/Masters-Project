# # import os
# # img_dir = "G:\\Shaft_Program_Files\\train"
# # print(sorted(os.listdir(img_dir)))
# # print(len(sorted(os.listdir(img_dir))))

# # txt = "welcome to the jungle"

# # x = txt.split('e')

# # print(x)
# img = {'object':[]}
# obj = {}
# obj['x0'] = float (90)
# obj['y0'] = float (91)
# obj['x1'] = float (94)
# obj['y1'] = float (95)
# img['object'] += [obj]
# obj={}
# obj['x0'] = float (20)
# obj['y0'] = float (21)
# obj['x1'] = float (24)
# obj['y1'] = float (25)
# img['object'] += [obj]

# print(img['object'])


# import cv2
# image = cv2.imread("G:\\Shaft_Program_Files\\testcv.png")
# img_channel = image[:,:,0]
# print(img_channel)

# import cv2
# import numpy as np

# a = np.zeros((100, 100,3))
# a[:,:,0] = 255

# b = np.zeros((100, 100,3))
# b[:,:,1] = 255

# c = np.zeros((100, 200,3)) 
# c[:,:,2] = 255

# # img = np.vstack((c, np.hstack((a, b))))
# img = a

# cv2.imshow('image', img)
# cv2.waitKey(0)
# import numpy as np

# image = np.zeros(3,3,dtype=float)

# for dim1 in range(3):
#     for dim2 in range(3):
#         print(image[dim1,dim2,0])
# import numpy as np

# mat = np.array ([[[1,2,12],[5,4,32],[300,200,34]],[[456,67,7564],[7,6,234],[8,9,43567]]])
# print(mat)
# print("\n")
# print(mat[:,:,::-1])
# print("\n")
# print(mat.shape)
# print("\n")

# math = np.expand_dims(mat,0)

# # print(math.shape)
# import numpy as np

# # arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]])

# # print(arr.shape)

# d = np.array([[[i + 2*j + 8*k for i in range(3)] for j in range(3)] for k in range(3)])
# print(d.shape)
# print("\n")
# print(d)
# print("\n")
# print(d[...,0])

# def yieldPrac(self):
#     for i in range (10):
#         x = 1 
#         yield print("The value of x {}".format(x))
#         x = x+i

# yieldPrac()

class Dog(object):
    def __init__(self,name):
        print (name)
    def bark(self,permit=True):
        print("Woof Woof")

Dobermann = Dog("Mahen")
animal_call = Dobermann.bark(True)
animal_call
animal_call
