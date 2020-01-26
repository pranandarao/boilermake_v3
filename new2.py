import cv2
import numpy as np
import copy
import math
import pyautogui as pag
# from appscript import app
from scipy.spatial import distance as dist
from imutils import face_utils
import argparse
import dlib


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
                help="path to input video file")
ap.add_argument("-t", "--threshold", type=float, default=0.27,
                help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type=int, default=2,
                help="the number of consecutive frames the eye must be below the threshold")

# parameters
cap_region_x_begin = 0  # start point/total width
cap_region_y_end = 1  # start point/total width
threshold = 60  # BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
mouse_position = [0, 0]

pag.FAILSAFE = False

# variables
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works
mouseSwitch = False

img = cv2.imread('resources\\CamMouse.PNG', 0)
cv2.imshow('Logo', img)


def printThreshold(thr):
    print("! Changed threshold to " + str(thr))


def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


def calculateFingers(res, drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0


# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)
cv2.namedWindow('trackbar')
cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)

while camera.isOpened():
    try:
        args = vars(ap.parse_args())
        EYE_AR_THRESH = args['threshold']
        EYE_AR_CONSEC_FRAMES = args['frames']

        # initialize the frame counters and the total number of blinks
        COUNTER = 0
        TOTAL = 0

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(args["shape_predictor"])

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        ret, frame = camera.read()
        threshold = cv2.getTrackbarPos('trh1', 'trackbar')
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                    pag.leftClick(mouse_position[0], mouse_position[1])
                    print('LEFT CLICK')

                # reset the eye frame counter
                COUNTER = 0

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        #  Main operation
        if isBgCaptured == 1:  # this part wont run until background captured
            img = removeBG(frame)
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            # cv2.imshow('mask', img)

            # convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            # cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            # cv2.imshow('ori', thresh)

            # get the coutours
            thresh1 = copy.deepcopy(thresh)
            contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length = len(contours)
            maxArea = -1
            if length > 0:
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i

                res = contours[ci]
                hull = cv2.convexHull(res)
                drawing = np.zeros(img.shape, np.uint8)
                # print([res])
                # print(type(res))
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

                point = np.mean(res, axis=0)
                # print(type(point))
                # print(point)
                # print(point[0][0])

                point1 = (int(point[0][0]), int(point[0][1]))
                # print(type(point))
                # print(point)
                # print(point.len())
                max_y = 0
                max_x = 0

                for i in res[0]:
                    # print(i[1], i[0])
                    if max_y < i[1]:
                        max_y = i[1]
                    if max_x < i[0]:
                        max_x = i[0]

                cv2.circle(drawing, point1, 10, (0, 0, 255))
                cv2.circle(drawing, (int(max_x), int(max_y)), 10, (0, 0, 255))

                screen_size = pag.size()

                mouse_position = [int(max_x), int(max_y)]
                mouse_position[1] -= 70

                mouse_position[0] *= screen_size[0] / 610
                if (mouse_position[0] > screen_size[0] - 1):
                    mouse_position[0] = screen_size[0] - 1
                mouse_position[1] *= screen_size[1] / 270
                if (mouse_position[1] > screen_size[1] - 1):
                    mouse_position[1] = screen_size[1] - 1

                isFinishCal, cnt = calculateFingers(res, drawing)

                # time.sleep(1)
                if (mouseSwitch):
                    pag.moveTo(mouse_position[0], mouse_position[1])

                    if (cnt > 1 and cnt < 4):
                        pag.leftClick(mouse_position[0], mouse_position[1])
                        print('LEFT CLICK')
                    elif (cnt >= 4):
                        pag.rightClick(mouse_position[0], mouse_position[1])
                        print('RIGHT CLICK')

            cv2.imshow('output', drawing)

        # Keyboard OP
        k = cv2.waitKey(10)
        if k == 27:  # press ESC to exit
            camera.release()
            cv2.destroyAllWindows()
            break
        elif k == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print('Background Captured')
        elif k == ord('r'):  # press 'r' to reset the background
            bgModel = None
            triggerSwitch = False
            isBgCaptured = 0
            print('Reset BackGround')
        elif k == ord('m'):  # press 'r' to reset the background
            mouseSwitch = True
            print('Activating Mouse')
    except AttributeError as error:
        print(error)
