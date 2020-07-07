import cv2
import numpy as np
import math

def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    kernel = np.ones((3, 3), np.uint8)
    skinMask=cv2.dilate(skinMask, kernel, iterations=1)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 100)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    #return cv2.cvtColor(skin, cv2.THRESH_BINARY_INV)
    return skin

def get_contours(roi, canvas):
    developer=np.zeros_like(roi)
    gray=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 120, 200)
    # filled = cv2.fillPoly(edged)

    _, binary = cv2.threshold(gray, 50, 255, 0)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        hull = cv2.convexHull(sorted_cnts[0], returnPoints=False)
        defects = cv2.convexityDefects(sorted_cnts[0], hull)

        cv2.drawContours(developer, sorted_cnts[0], -1, (0,255,0), 3)

        extLeft = tuple(sorted_cnts[0][sorted_cnts[0][:, :, 0].argmin()][0])
        extRight = tuple(sorted_cnts[0][sorted_cnts[0][:, :, 0].argmax()][0])
        extTop = tuple(sorted_cnts[0][sorted_cnts[0][:, :, 1].argmin()][0])
        extBot = tuple(sorted_cnts[0][sorted_cnts[0][:, :, 1].argmax()][0])

        cv2.circle(developer, extLeft, 8, (255, 255, 255), -1)
        # cv2.circle(frame, extRight, 8, (255, 255, 255), -1)
        cv2.circle(developer, extTop, 8, (255, 255, 255), -1)
        # cv2.circle(frame, extBot, 8, (255, 255, 255), -1)

        point_defect=[]

        if np.array(defects).any() != None:
            for i in range(np.array(defects).shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(sorted_cnts[0][s][0])
                end = tuple(sorted_cnts[0][e][0])
                far = tuple(sorted_cnts[0][f][0])

                #calculate triangles
                s_e = math.sqrt((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)
                e_f = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                f_s = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)

                s=(s_e + e_f + f_s)/2

                area = math.sqrt(s*(s - s_e)*(s - e_f)*(s - f_s))
                d = (2 * area) / s_e        #height = 2*area/base
                angle = math.acos((e_f ** 2 + f_s ** 2 - s_e ** 2) / (2 * e_f * f_s)) * 57

                if d>=25 and angle<=90:
                    cv2.circle(developer, far, 3, [0, 0, 255], -1)
                    point_defect.append(far)

                cv2.line(developer, start, end, [255, 0, 0], 2)

            flag =True
            if len(point_defect)>=3:
                flag= False

            else:
                for point in point_defect:
                    if point[0] < extTop[0] and point[1] > extLeft[1]:
                        flag = False

            if flag:
                cv2.circle(frame[20:400, 250:635], extTop, 8, (0, 0, 255), -1)
                cv2.circle(developer, extTop, 8, (0, 0, 255), -1)
                cv2.circle(canvas, extTop, 2, (0, 0, 255), 4)

            cv2.imshow("Developer's Window", developer)
            # canny_images = np.hstack((gray, edged, binary))
            # cv2.imshow("Test", canny_images)
            return canvas

    return np.zeros_like(roi)

####################################################################################################################

cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

cam = cv2.VideoCapture(0)
if cam.isOpened():
    ret, frame = cam.read()
else:
    ret = False
font = cv2.FONT_HERSHEY_SIMPLEX
canvas = np.zeros_like(frame[20:400, 250:635])

while ret:
    ret, frame = cam.read()
    frame=cv2.flip(frame,1)
    rectangle=cv2.rectangle(frame, (250, 20), (640, 480), (0, 255, 0), 3)
    cv2.putText(frame, 'Place hand in the box', (380, 15), font, 0.5, (0, 0, 255), 0, cv2.LINE_AA)
    roi=frame[20:400, 250:635]
    roi=extractSkin(roi)
    canvas = get_contours(roi, canvas)

    # roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    # roi=cv2.threshold(roi, 15, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow("Canvas", canvas)
    frame[20:400, 250:635] = cv2.addWeighted(frame[20:400, 250:635], 0.3, canvas, 0.7, 0.0)
    # cv2.imshow("ROI", roi)
    cv2.imshow("Camera", frame)
    # control = np.zeros_like(frame)
    # temp = np.hstack((frame, control))
    # cv2.imshow("Window", temp)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cam.release()