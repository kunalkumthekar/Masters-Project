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

for dim1 in range(128):
    for dim2 in range(128):
        print(image[dim1,dim2,0])
        