import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
fingerCoor = [(8,6), (12,10), (16,14), (20,18)]
thumbCoor = (4,2)

def getLabel(index, hand, results):
    output = None
    label = results.multi_handedness[index].classification[0].label
    coords = tuple(np.multiply(np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),[640,480]).astype(int))

    output = label, coords
    
    return output
 
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands = 2) as hands:
    cap = cv2.VideoCapture(0) #Cam feed
    while cap.isOpened():
        ret, img = cap.read()

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image = cv2.flip(image,1) # Flips image horizontally
        # Detections
        results = hands.process(image) # process method expects input image in RGB
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Make it BGR again to render the results properly
        
        # Render the results
        if results.multi_hand_landmarks:
            rightHandCoor=[] 
            leftHandCoor=[]
            leftFingers=0
            rightFingers=0
            
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

                if getLabel(num, hand, results):
                    label, coord = getLabel(num, hand, results)
                    cv2.putText(image, label, coord, cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

                    for landmark_id, landmark in enumerate(hand.landmark):
                        height, width, color_channel = img.shape
                        cx, cy = int(landmark.x * width), int(landmark.y * height) # This calculates the pixel coordinates of each landmark

                        if label == "Left":
                            leftHandCoor.append((cx,cy))
                        elif label == "Right":
                            rightHandCoor.append((cx,cy))
                    
                    if label == "Left":
                        for coors in fingerCoor:
                            if leftHandCoor[coors[0]][1] < leftHandCoor[coors[1]][1]:
                                leftFingers += 1
                        if leftHandCoor[thumbCoor[0]][0] > leftHandCoor[thumbCoor[1]][0]:
                            leftFingers += 1
                        cv2.putText(image, str(leftFingers), (50,150), cv2.FONT_HERSHEY_PLAIN, 6, (255,0,0), 6)

                    if label == "Right":
                        for coors in fingerCoor:
                            if rightHandCoor[coors[0]][1] < rightHandCoor[coors[1]][1]:
                                rightFingers += 1
                        if rightHandCoor[thumbCoor[0]][0] < rightHandCoor[thumbCoor[1]][0]:
                            rightFingers += 1 
                        cv2.putText(image, str(rightFingers), (350,150), cv2.FONT_HERSHEY_PLAIN, 6, (255,0,0), 6)

            cv2.putText(image, "Total Fingers: " + str(leftFingers + rightFingers), (100,450), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
        cv2.imshow("Finger Counter", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


   

    