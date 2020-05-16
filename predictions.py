from cnn_model import define_model, model_weights
import cv2
import os.path
import numpy as np

def prediction_path(path):
    
    model = define_model()
    model = model_weights(model)
    
    # list of given emotions
    EMOTIONS = ['Angry', 'Disgusted', 'Fearful',
                'Happy', 'Sad', 'Surprised', 'Neutral']

    if os.path.exists(path):
        
        img = cv2.imread(path, 0)
        
        if img is None:
            print('Invalid image !!')
            return 
    else:
        print('Image not found')
        return
    
    img = cv2.resize(img, (48, 48))
    img = np.reshape(img, (1, 48, 48, 1))
    # do prediction
    result = model.predict(img)

    print('Detected emotion: ' + str(EMOTIONS[np.argmax(result[0])]))
    
    return