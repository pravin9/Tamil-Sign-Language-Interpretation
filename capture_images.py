import cv2
import os
import xlrd

TAMIL_SIGN_LANGUAGE_HOME = os.path.dirname(os.path.realpath(__file__))
# SIGN_IMAGE_UPLOAD_FOLDER = 'C:/Users/User/Downloads/Code Implementation/uploads'.format(TAMIL_SIGN_LANGUAGE_HOME)
SIGN_IMAGE_UPLOAD_FOLDER = 'C:/Users/User/Downloads/Flask/uploads'


# function to capture frames from the video
def getvideoframes(path):
    framecapture = cv2.VideoCapture(path)
    i = 0
    while framecapture.isOpened():
        # capture frames from the video.
        # ret determines if the video is captured, frame is the captured image
        ret, frame = framecapture.read()
        # if ret is false no video was captured
        if not ret:
            break
            # captures frames each second
        if i % 60 == 0:
            cv2.imwrite(os.path.join(SIGN_IMAGE_UPLOAD_FOLDER, str(i) + '.jpg'), frame)
        i += 1
    framecapture.release()
    cv2.destroyAllWindows()


# function to read the tamil word mapping xls file
def conversion():
    #workbook = xlrd.open_workbook(os.path.join(os.getcwd(), 'Classes.xlsx'))
    workbook = xlrd.open_workbook('C:/Users/User/Downloads/Flask/Classes.xlsx')
    sheet = workbook.sheet_by_index(0)

    col_a = sheet.col_values(0, 1)
    col_b = sheet.col_values(1, 1)

    my_dict = {a: b for a, b in zip(col_a, col_b)}
    return my_dict


# function to map the tamil word and return the tamil for the detected sign
def getresult(my_dict, emotion):
    result = ""
    for extracted_key in emotion:
        for keys in my_dict.keys():
            if keys == extracted_key:
                value = my_dict.get(keys)
                result += value + " "
                break
    return result
