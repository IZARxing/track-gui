# coding: utf-8
import pyautogui
import cv2
import numpy as np


def get_cord_real_time():
    """get mouse coordinate in real time"""
    print('mouse coordinate now: ')
    try:
        while 1:
            x, y = pyautogui.position()
            print_position = f'X: {str(x).rjust(5)}, Y: {str(y).rjust(5)}'
            print(f"\r{print_position}", end='', flush=True)
    except KeyboardInterrupt:
        print('\n')


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    3.0, (255, 0, 0), thickness=5)
        cv2.imshow("image", img)


if __name__ == '__main__':
    # get_cord_real_time()
    cap = cv2.VideoCapture("./data/videos/MOT16-03.mp4")  # 选你要检测的视频，读取帧设置计数区域，获得点的坐标， 这里改成0就是读摄像头
    _, img = cap.read()

    # w, h, _ = img.shape
    # img_resize = cv2.resize(img, (int(w/2), int(h/2)))
    # cv2.imwrite('./chaochang.jpg', img_resize)

    # print img.shape
    cv2.namedWindow("image", cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)

    while True:
        try:
            cv2.waitKey(100)
        except Exception:
            cv2.destroyWindow("image")
            break

    cv2.waitKey(0)
    cv2.destroyAllWindow()
