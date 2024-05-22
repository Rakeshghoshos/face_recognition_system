import cv2
import os

def generate_dataset(name):
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    directory = f"data/{name}"
    os.mkdir(directory)
    def cropped(img):
        grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(grey_img,1.3,5)

        if face is ():
            return None

        for (x,y,w,h) in face:
            croped_face = img[y:y+h,x:x+w]
        return croped_face
    
    cap = cv2.VideoCapture(0)
    id = 0

    while True:
        ret, frame = cap.read()
        if cropped(frame) is not None:
            id+=1
            face =cv2.resize(cropped(frame),(300,300))
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            file_path = f"data/{name}/{str(id)}.jpg"
            cv2.imwrite(file_path,face)
            cv2.putText(face,str(id),(140,50),cv2.FONT_HERSHEY_COMPLEX,1,(90,150,150),2)
            cv2.imshow("croped face",face)
            if cv2.waitKey(1) == 13 or int(id) == 1000:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("collection completed")
