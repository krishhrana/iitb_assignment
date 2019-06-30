import cv2
import tensorflow as tf

CATEGORIES = ["Car", "Auto", "Motorcycle"]
def prepare(filepath):
    IMG_SIZE = 70  # 
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model=tf.keras.models.load_model("257x32.CNN.model") #loading the model
pred = model.predict([prepare("/Users/krishrana/Downloads/cars4.jpg")])

print (pred) #predicts the image and gives o/p in binary. 
#(1,0,0)----> car
#(0,1,0)----> auto
#(0,0,1)----> motorcycle

