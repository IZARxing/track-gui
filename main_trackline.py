from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win  import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import sys
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
# LoadWebcam 的最后一个返回值改为 self.cap
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box, plot_one_box_PIL, compute_color_for_id, distancing
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.capnums import Camera
from dialog.rtsp_win import Window

# deepsort
import tracker

flag = False

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # 发送信号：正在检测/暂停/停止/检测结束/错误报告
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './best_person_optimize.pt'            # 设置权重
        self.current_weight = './best_person_optimize.pt'     # 当前权重
        self.source = '0'                       # 视频源
        self.conf_thres = 0.25                  # 置信度
        self.iou_thres = 0.45                   # iou
        self.jump_out = False                   # 跳出循环
        self.is_continue = True                 # 继续/暂停
        self.percent_length = 1000              # 进度条
        self.rate_check = True                  # 是否启用延时
        self.rate = 100                         # 延时HZ

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=True,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3, [0, 1, 2, 3, 5, 6, 7]
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            ##########################################################################################################
            # 设置计数撞线区域
            if self.source.endswith(".mp4") or self.source.endswith(".avi"):
                cap = cv2.VideoCapture(self.source)
                _, img = cap.read()
                img = cv2.resize(img, (1920, 1080))
                imgh, imgw, imgc = img.shape
                cap.release()
            elif self.source.isdigit():
                cap = cv2.VideoCapture(int(self.source))
                _, img = cap.read()
                img = cv2.resize(img, (1920, 1080))
                imgh, imgw, imgc = img.shape
                cap.release()
            else:
                cap = cv2.VideoCapture(self.source)
                _, img = cap.read()
                #img = cv2.imdecode(np.fromfile(self.source, dtype=np.uint8), 1)
                img = cv2.resize(img, (1920, 1080))
                imgh, imgw, imgc = img.shape
                cap.release()


            # print(imgh, imgw)
            # 根据视频尺寸，填充一个polygon，供撞线计算使用
            mask_image_temp = np.zeros((imgh, imgw), dtype=np.uint8)

            # 初始化2个撞线polygon

            # list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379],
            #                  [604, 437],
            #                  [299, 375], [267, 289]]
            # list_pts_blue = [[216, 444], [469, 496], [822, 571], [1247, 655], [1786, 765],
            #                  [1786, 795], [1287, 691], [874, 600], [499, 525], [179, 461]]
            # list_pts_blue = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            #                  [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            list_pts_blue = [[90, 483], [357, 551], [774, 655], [1269, 791], [1734, 908],
                             [1719, 937], [1284, 830], [796, 701], [376, 597], [60, 509]]   #稀疏街道1
            # list_pts_blue = [[3, 337], [403, 334], [1046, 332], [1604, 329], [1914, 332],
            #                  [1914, 378], [1642, 378], [1146, 381], [468, 389], [0, 386]]  #密集街道1
            ndarray_pts_blue = np.array(list_pts_blue, np.int32)
            polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
            polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

            # 填充第二个polygon
            mask_image_temp = np.zeros((imgh, imgw), dtype=np.uint8)
            # list_pts_red = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
            #                    [594, 637], [118, 483], [109, 303]]
            # list_pts_red = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
            #                  [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            list_pts_red = [[45, 519], [417, 613], [845, 723], [1325, 846], [1719, 954],
                               [1700, 992], [1295, 882], [833, 762], [376, 642], [0, 538]]#稀疏街道1

            # list_pts_red = [[9, 403], [514, 400], [1216, 397], [1724, 397], [1900, 406],
            #                 [1911, 452], [1707, 452], [1239, 452], [582, 450], [18, 452]]#密集街道1
            ndarray_pts_red = np.array(list_pts_red, np.int32)
            polygon_red_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_red], color=2)
            polygon_red_value_2 = polygon_red_value_2[:, :, np.newaxis]

            # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
            polygon_mask_blue_and_red = polygon_blue_value_1 + polygon_red_value_2

            # 缩小尺寸，1920x1080->960x540
            polygon_mask_blue_and_red = cv2.resize(polygon_mask_blue_and_red, (imgw, imgh))

            # 蓝 色盘 b,g,r
            blue_color_plate = [255, 0, 0]
            # 蓝 polygon图片
            blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

            # 红 色盘
            red_color_plate = [127, 0, 255]
            # 红 polygon图片
            red_image = np.array(polygon_red_value_2 * red_color_plate, np.uint8)

            # 彩色图片（值范围 0-255）
            color_polygons_image = blue_image + red_image
            # 缩小尺寸，1920x1080->960x540
            color_polygons_image = cv2.resize(color_polygons_image, (imgw, imgh))

            # list 与蓝色polygon重叠
            list_overlapping_blue_polygon = []

            # list 与红色polygon重叠
            list_overlapping_red_polygon = []

            # 进入数量
            down_count = 0
            # 离开数量
            up_count = 0

            font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
            draw_text_postion = (int(imgw * 0.01), int(imgh * 0.05))
            #####################################################################################################

            # 初始化yolov5模型
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            print(names)
            if half:
                model.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0
            # 跳帧检测
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)

            # 双向过线计数字典初始化
            statistic_dic = {"UP COUNT": 0, "DOWN COUNT": 0, "上行ID": 0, "下行ID": 0}
            # 定义跟踪id、point缓存空字典，{key：value}
            dict_box = dict()
            while True:
                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('停止')
                    break
                # 临时更换模型
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                # 暂停开关
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    im0s = cv2.resize(im0s, (1920, 1080))
                    # print(im0s.shape)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    # 每三十帧刷新一次输出帧率
                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    # statistic_dic = {name: 0 for name in names}
                    # statistic_dic = {"UP COUNT": 0, "DOWN COUNT": 0}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)

                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()

                        list_bboxs = []
                        bboxes = []
                        confs = []
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                # statistic_dic[names[c]] += 1
                                # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                # # im0 = plot_one_box_PIL(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)  # 中文标签画框，但是耗时会增加
                                # plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                #              line_thickness=line_thickness)
                                lbl = names[int(cls)]
                                if lbl not in ["Person"]:  # 根据你的模型类别，对目标进行筛选，不在列表中不进行计数和跟踪
                                    continue
                                pass
                                x1, y1 = int(xyxy[0]), int(xyxy[1])
                                x2, y2 = int(xyxy[2]), int(xyxy[3])
                                bboxes.append(
                                    (x1, y1, x2, y2, lbl, conf))

                            confs = det[:, 4]

                        if len(bboxes) > 0:
                            # 将检测结果传入deepsort
                            list_bboxs = tracker.update(bboxes, im0)
                            # draw boxes for visualization, list_bboxs=[[xmin,ymin,xmax,ymax,class,id]]
                            if len(list_bboxs) > 0:
                                for j, (output, conf) in enumerate(zip(list_bboxs, confs)):
                                    bboxes = output[0:4]
                                    id = output[5]
                                    cls_name = output[4]
                                    # c = int(cls)  # integer class
                                    label = f'{id} {cls_name} {conf:.2f}'
                                    color = compute_color_for_id(id)
                                    plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)

                            # 绘制轨迹模块
                            ###################################################################
                            if len(list_bboxs) > 0:
                                track_inf = []
                                for obj_coods in list_bboxs:  # 对deepsort输出的内容进行筛选，拿到坐标和id
                                    track_inf.append([obj_coods[0], obj_coods[1], obj_coods[2],
                                                      obj_coods[3], obj_coods[5]])
                                if len(track_inf) > 0:
                                    np_outputs = np.stack(np.array(track_inf), axis=0)  # 将track_inf列表转矩阵并进行纵向堆叠
                                    bbox_xyxy = np_outputs[:, 0:4]  # 取出目标坐标
                                    identities = np_outputs[:, 4]  # 取出识别到的id
                                    # top left x, top left y,  w and h
                                    box_xywh = xyxy2xywh(bbox_xyxy)  # x1, y1, x2, y2 ——> x1, y1, w, h

                                    for j in range(len(box_xywh)):
                                        x_center = box_xywh[j][0]  # + box_xywh[j][2] / 2
                                        y_center = box_xywh[j][1]  # + box_xywh[j][3] / 2
                                        obj_id = list_bboxs[j][5]
                                        # center = [x_center, y_center]
                                        center = [x_center, y_center, list_bboxs[j]]
                                        # print(center)
                                        dict_box.setdefault(obj_id, []).append(center)

                                    # 绘制不同目标id的轨迹
                                    for key, value in dict_box.items():
                                        if len(value) < 2:
                                            continue
                                        if not key in list(identities):
                                            # dict_box.pop(key)
                                            continue
                                        # print(key, value)
                                        color = compute_color_for_id(int(key))

                                        # 历史轨迹
                                        for a in range(len(value) - 1):
                                            index_start = a
                                            index_end = index_start + 1
                                            # map(int,"1234")转换为list[1,2,3,4]
                                            cv2.line(im0, tuple(map(int, value[index_start][0:2])),
                                                     tuple(map(int, value[index_end][0:2])),
                                                     color, thickness=2, lineType=8)
                            ############################################################

                            # 画框
                            # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                            output_image_frame = tracker.draw_bboxes(im0, list_bboxs, line_thickness=None)
                            pass
                        else:
                            # 如果画面中 没有bbox
                            output_image_frame = im0
                        pass

                        # 输出图片
                        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

                        if len(list_bboxs) > 0:
                            # ----------------------判断撞线----------------------
                            for item_bbox in list_bboxs:
                                x1, y1, x2, y2, label_name, track_id = item_bbox

                                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                                y1_offset = int(y1 + ((y2 - y1) * 1))

                                # 撞线的点
                                y = y1_offset
                                x = x1
                                # print(polygon_mask_blue_and_red[y, x])
                                if polygon_mask_blue_and_red[y, x] == 1:
                                    # 如果撞 蓝polygon
                                    if track_id not in list_overlapping_blue_polygon:
                                        list_overlapping_blue_polygon.append(track_id)
                                    pass


                                    # 判断 红polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 外出方向
                                    if track_id in list_overlapping_red_polygon:
                                        # 外出+1
                                        up_count += 1
                                        statistic_dic["UP COUNT"] = up_count
                                        print(
                                            f'类别: {label_name} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_red_polygon}')
                                        statistic_dic["上行ID"] = track_id
                                        # 删除 红polygon list 中的此id
                                        list_overlapping_red_polygon.remove(track_id)

                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass

                                elif polygon_mask_blue_and_red[y, x] == 2:
                                    # 如果撞 红polygon
                                    if track_id not in list_overlapping_red_polygon:
                                        list_overlapping_red_polygon.append(track_id)
                                    pass

                                    # 判断 蓝polygon list 里是否有此 track_id
                                    # 有此 track_id，则 认为是 进入方向
                                    if track_id in list_overlapping_blue_polygon:
                                        # 进入+1
                                        down_count += 1
                                        statistic_dic["DOWN COUNT"] = down_count

                                        print(
                                            f'类别: {label_name} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')

                                        # 删除 蓝polygon list 中的此id
                                        statistic_dic["下行ID"] = track_id
                                        list_overlapping_blue_polygon.remove(track_id)

                                        pass
                                    else:
                                        # 无此 track_id，不做其他操作
                                        pass
                                    pass
                                else:
                                    pass
                                pass

                            pass

                            # ----------------------清除无用id----------------------
                            list_overlapping_all = list_overlapping_red_polygon + list_overlapping_blue_polygon
                            for id1 in list_overlapping_all:
                                is_found = False
                                for _, _, _, _, _, bbox_id in list_bboxs:
                                    if bbox_id == id1:
                                        is_found = True
                                        break
                                    pass
                                pass

                                if not is_found:
                                    # 如果没找到，删除id
                                    if id1 in list_overlapping_red_polygon:
                                        list_overlapping_red_polygon.remove(id1)
                                    pass
                                    if id1 in list_overlapping_blue_polygon:
                                        list_overlapping_blue_polygon.remove(id1)
                                    pass
                                pass
                            list_overlapping_all.clear()
                            pass

                            # 清空list
                            list_bboxs.clear()

                            pass
                        else:
                            # 如果图像中没有任何的bbox，则清空list
                            list_overlapping_blue_polygon.clear()
                            list_overlapping_red_polygon.clear()
                            pass
                        pass

                        text_draw = 'DOWN: ' + str(down_count) + \
                                    ' , UP: ' + str(up_count)
                        im0 = cv2.putText(img=output_image_frame, text=text_draw,
                                                         org=draw_text_postion,
                                                         fontFace=font_draw_number,
                                                         fontScale=1, color=(0, 0, 0), thickness=2)


                    # 控制视频发送频率
                    if self.rate_check:
                        time.sleep(1/self.rate)
                    # print(type(im0s))
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('检测结束')
                        # 正常跳出循环
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)


class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        self.setWindowFlags(Qt.CustomizeWindowHint)
        # self.setWindowFlags(Qt.FramelessWindowHint)
        # 自定义标题栏按钮
        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        self.closeButton.clicked.connect(self.close)

        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5线程
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type           # 权重
        self.det_thread.source = '0'                                    # 默认打开本机摄像头，无需保存到配置文件
        if flag == True :
            self.det_thread.source = '1'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)
        self.maButton.clicked.connect(self.chose_ma)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        # self.comboBox.currentTextChanged.connect(lambda x: self.statistic_msg('模型切换为%s' % x))
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def checkrate(self):
        if self.checkBox.isChecked():
            # 选中时
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        flag = True
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:1234@192.168.43.223:8554/live"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='提示', text='\n 正在加载rtsp视频流，请稍等', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('加载rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):
        try:
            self.stop()
            # MessageBox的作用：留出2秒，让上一次摄像头安全release
            MessageBox(
                self.closeButton, title='提示', text='\n正在检测摄像头设备,请稍等！', time=2000, auto=True).exec_()
            # 自动检测本机有哪些摄像头
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('加载摄像头：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    # 导入配置文件
    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            new_config = {"iou": 0.26,
                          "conf": 0.33,
                          "rate": 10,
                          "check": 0
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            iou = config['iou']
            conf = config['conf']
            rate = config['rate']
            check = config['check']
        self.confSpinBox.setValue(iou)
        self.iouSpinBox.setValue(conf)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)   # 3秒后自动清除

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('模型切换为%s' % x)

    def open_file(self):
        # source = QFileDialog.getOpenFileName(self, '选取视频或图片', os.getcwd(), "Pic File(*.mp4 *.mkv *.avi *.flv "
        #                                                                    "*.jpg *.png)")
        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, '选取视频或图片', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('加载文件：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            # 切换文件后，上一次检测停止
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    # 继续/暂停
    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = '摄像头设备' if source.isnumeric() else source

            self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('暂停')

    # 退出检测循环
    def stop(self):
        self.det_thread.jump_out = True

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            # QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        # self.setCursor(QCursor(Qt.ArrowCursor))

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 实时统计
    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        # 如果摄像头开着，先把摄像头关了再退出，否则极大可能可能导致检测线程未退出
        self.det_thread.jump_out = True
        # 退出时，保存设置
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='提示', text='\n 正在关闭行人检测和跟踪系统，请稍等！', time=2000, auto=True).exec_()
        sys.exit(0)

    def chose_ma(self):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='关于', text='\n 版本信息：行人检测和跟踪系统 version1.0 \n 更新网址：https：//www.xxx.com \n 作者联系方式：xxx@163.com', auto=False).exec_()

            self.statistic_msg('查看版本信息：{}')
        except Exception as e:
            self.statistic_msg('%s' % e)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
