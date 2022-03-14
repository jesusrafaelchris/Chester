from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io
import os
import os.path

img_width, img_height = 70,70
model = load_model('/Users/christiangrinling/Desktop/Keras/model9.h5') # Load the model

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])

path = '/Users/christiangrinling/Desktop/Keras/images1/img' #path to extracted images
list = os.listdir(path) # dir is your directory path
number_files = len(list)


y = 0
for i in range (y,65):
    if os.path.isfile(path + (str(i)) + '.jpg'):
        img_path = path + (str(i)) + '.jpg'

        test_image= image.load_img(img_path, target_size = (img_width, img_height, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        images = np.vstack([test_image])
        result = model.predict_classes(images)

        if result == 9:                             #change these paths
            os.rename(path + (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/redking/img' + (str(i)) + '.jpg')
        elif result == 11:
            os.rename(path + (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/redpawn/img' + (str(i)) + '.jpg')
        elif result == 12:
            os.rename(path + (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/redqueen/img' + (str(i)) + '.jpg')
        elif result == 10:
            os.rename(path+ (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/redknight/img' + (str(i)) + '.jpg')
        elif result == 8:
            os.rename(path+ (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/redcastle/img' + (str(i)) + '.jpg')
        elif result == 7:
            os.rename(path+ (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/redbishop/img' + (str(i)) + '.jpg')
        elif result == 0:
            os.rename(path+ (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/empty/img' + (str(i)) + '.jpg')
        elif result == 11:
            os.rename(path + (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/greenpawn/img' + (str(i)) + '.jpg')
        elif result == 12:
            os.rename(path + (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/greenqueen/img' + (str(i)) + '.jpg')
        elif result == 10:
            os.rename(path+ (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/greenknight/img' + (str(i)) + '.jpg')
        elif result == 8:
            os.rename(path + (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/greencastle/img' + (str(i)) + '.jpg')
        elif result == 7:
            os.rename(path + (str(i)) + '.jpg', '/Users/christiangrinling/Desktop/greenbishop/img' + (str(i)) + '.jpg')

else:
y += 1

#'train_empty_dir': 0,
#'train_green_bishop_dir': 1,
#'train_green_castle_dir': 2,
#'train_green_king_dir': 3,
#'train_green_knight_dir': 4,
#'train_green_pawn_dir': 5,
#'train_green_queen_dir': 6,

#'train_red_bishop_dir': 7,
#'train_red_castle_dir': 8,
#'train_red_king_dir ': 9,
#'train_red_knight_dir': 10,
#'train_red_pawn_dir': 11,
#'train_red_queen_dir': 12

#Epoch 30/30
#100/100 [==============================] - 7s 72ms/step - loss: 0.1444 - acc: 0.9560 - val_loss: 0.1078 - val_acc: 0.9680
