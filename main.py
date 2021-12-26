import cv2
import mediapipe as mp

cap =cv2.VideoCapture(0)
mpFaceMesh= mp.solutions.face_mesh
facemesh=mpFaceMesh.FaceMesh()
mpdrow= mp.solutions.drawing_utils

while True :
    succes ,img =cap.read()
    imgRGb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result= facemesh.process(imgRGb)
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            mpdrow.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS )




    cv2.imshow("Face Mesh", img)
    cv2.waitKey(1)