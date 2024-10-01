from deepface import DeepFace 
import cv2

cam = cv2.VideoCapture(0)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

face_emotion='none'
while True:
    ret, frame = cam.read()
    if not ret:
        break
    # Display the captured frame
    
    wk=cv2.waitKey(1)
    # Press 'q' to exit the looprr

    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (0, 120, 255)
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.putText() method

    try:
        result = DeepFace.analyze(frame, actions=['emotion'],detector_backend="mtcnn")

        sorted_emotions = sorted(result[0]['emotion'].items(), key=lambda item: item[1], reverse=True)

        # Display the sorted emotions
        print(sorted_emotions)
        print("dominant:")
        print(sorted_emotions[0][0])
        face_emotion=result[0]['dominant_emotion']
        if result[0]['dominant_emotion']=='neutral':
            if (sorted_emotions[0][1]-sorted_emotions[1][1])<20:
                face_emotion=sorted_emotions[1][0]

        
       
        print(result[0])





        #print(result['dominant_emotion'])
      

    except ValueError as e:
        face_emotion="No Detected Face"
        print("no face detected")

        # Optionally, save the image for inspection
        



    if wk  == ord('q'):
        break

    processedImage = cv2.putText(frame, face_emotion, org, font, 
                fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Camera', processedImage )
cam.release()
cv2.destroyAllWindows()