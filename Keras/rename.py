
import os


i = 1

for filename in os.listdir('/Users/christiangrinling/Desktop/Keras/extracted/'):
    dst ="img" + str(i) + ".jpg"
    src ='/Users/christiangrinling/Desktop/Keras/extracted/'+ filename
    dst ='/Users/christiangrinling/Desktop/Keras/extracted/'+ dst
    os.rename(src, dst)
    i += 1
