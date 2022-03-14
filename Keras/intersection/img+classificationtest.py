from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

model = load_model('/Users/christiangrinling/Desktop/Keras/model1.h5')
img_width, img_height = 224,224

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
result1 = []
for i in range (0,21):
    img_path = '/Users/christiangrinling/Desktop/Keras/extracted/img' + (str(i)) + '.jpg'

    test_image= image.load_img(img_path, target_size = (img_width, img_height, 3))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    images = np.vstack([test_image])

    result = model.predict_classes(images)
    result1 = np.append(result1, result)
    result1 = result1.astype(int)
    result2 = model.predict_proba(images)
    print(np.around(result2,3))
print(result1)
"""if result == [0]:
    result = 'King'

elif result == [1]:
    result = 'Knight'

elif result == [2]:
     result = 'Pawn'

elif result == [3]:
     result = 'Queen'

else:
    print('Not found')"""

chessboard = np.zeros(shape = (8,8), dtype = 'int')

coords = np.array([[2,3], [7,8], [3,4]])


x1,y1= coords[0][0], coords[0][1]
x2, y2 = [1,2]
x3, y3 = [6,1]
x4, y4 = [4,1]
x5, y5 = [7,5]
x6, y6 = [2,2]
x7, y7 = [3,7]
#........
x64,y64 = [5,5]


for i in range(8):
    for j in range(8):
        if x1 == i and y1 == j:
            column[i][j]=result1[0]

for i in range(8):
    for j in range(8):
        if x2 == i and y2 == j:
            column[i][j]=result1[1]

for i in range(8):
    for j in range(8):
        if x3 == i and y3 == j:
            column[i][j]=result1[2]



for i in range(8):
    for j in range(8):
        if x5 == i and y5 == j:
            column[i][j]=result1[4]


for i in range(8):
    for j in range(8):
        if x6 == i and y6 == j:
            column[i][j]=result1[5]
print(column)
