import cv2
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model('baseline')

emotion = ["angry","disgust","fear","happy","sad","surprised","neutral"]
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(48, 48),
    tf.keras.layers.Rescaling(1./255)
])

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_default.xml')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = frame[:,::-1,:]
    frame= frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    face = face_cascade.detectMultiScale(gray,1.1,3)  
    img = frame
    if(len(face)>=1):
        (x,y,w,h)= face[0]
        x -= 20
        y -= 20
        w += 20
        h += 20
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        img = frame[:][y:y+h,x:x+w]  
    
    img = Image.fromarray(img)
    img = resize_and_rescale(img)
    img = tf.math.reduce_mean(img,axis=-1,keepdims=True)
    img = tf.expand_dims(img, axis=0)
    pre = model.predict(img)
    pre = tf.math.argmax(pre, axis=-1)
    frame = cv2.putText(frame, emotion[pre.numpy().item()], (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,112,67), 2)

    cv2.imshow('emotion', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()