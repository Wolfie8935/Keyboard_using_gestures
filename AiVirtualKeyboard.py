import cv2
import cvzone
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from time import sleep
from pynput.keyboard import Controller

# url = 'http://10.3.186.117:8080//video'

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
keys = [["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
        ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = []
clicked = False

keyboard = Controller()

def drawAll(img, buttonList):

    # for button in buttonList:
    #     x, y = button.pos
    #     w, h = button.size
    #     cv2.rectangle(img, button.pos, (x + w, y + h), (60, 60, 60), cv2.FILLED)
    #     cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN,
    #                 4, (255, 255, 255), 4)
    # return img

    imgNew = np.zeros_like(img, dtype=np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]), 20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x+button.size[0], y+button.size[1]),(60,60,60), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x+40, y+60), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    # print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]
    return out

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.text = text
        self.size = size


buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bboxInfo = detector.findPosition(img)  # cvzone = 1.4.1  mediapipe = 0.8.7.2
    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][0] < x+w and y < lmList[8][1] < y+h:
                cv2.rectangle(img, button.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN,
                            4, (255, 255, 255), 4)
                l, _, _ = detector.findDistance(8, 12, img)
                # print(l)

                # when clicked
                if l < 20:
                    keyboard.press(button.text)
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN,
                                4, (255, 255, 255), 4)
                    finalText.append(button.text)
                    clicked = True
                    sleep(0.3)

    if not lmList:
        clicked = False

    cv2.rectangle(img, (50, 650), (700, 500), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, "".join(finalText), (60, 625), cv2.FONT_HERSHEY_PLAIN,
                5, (255, 255, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)