import cv2 as cv2
import numpy as np

def load_images():
    train_images = []
    for name in range(28710):
        image = cv2.imread('C:/Users/KIIT/Desktop/emotion_detect/data/dataset/Training/'+ str(name) +'.jpg',0)
        if image is not None:
            train_images.append(image)
            
    validation_images = []
    for name in range(28710,32299):
        image = cv2.imread('C:/Users/KIIT/Desktop/emotion_detect/data/dataset/PublicTest/'+str(name)+'.jpg',0)
        if image is not None:
            validation_images.append(image)

    test_images = []
    for name in range(32299,35888):
        image = cv2.imread('C:/Users/KIIT/Desktop/emotion_detect/data/dataset/PrivateTest/'+str(name)+'.jpg',0)
        if image is not None:
            test_images.append(image)
    
    return train_images,validation_images,test_images

train_images, validation_images, test_images = load_images()

train_images=np.array(train_images)
validation_images=np.array(validation_images)
test_images = np.array(test_images)

np.save('C:/Users/KIIT/Desktop/emotion_detect/data/dataset/train_images1.npy', train_images)
np.save('C:/Users/KIIT/Desktop/emotion_detect/data/dataset/validation_images1.npy', validation_images)
np.save('C:/Users/KIIT/Desktop/emotion_detect/data/dataset/test_images1.npy', test_images)