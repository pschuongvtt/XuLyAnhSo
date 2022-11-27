#dnn : Deep neural network 
#Phát hiện gương mặt
#opencv_face_detector : file config 
#models : file model hướng dẫn sử dụng 
#Thư mục lấy dữ liệu D:\opencv\sources\samples\dnn
#link dowload git : https://github.com/spmallick/learnopencv/blob/master/AgeGender/opencv_face_detector_uint8.pb
'''
Cách 1: Training = Tenserflow
open cho 2 file 
+ model.pb
+ config.pbtxt
link dowload git : https://github.com/spmallick/learnopencv/blob/master/AgeGender/opencv_face_detector_uint8.pb
Lệnh cmd : python object_detection.py --model=opencv_face_detector_uint8.pb --config=opencv_face_detector.pbtxt --width=300 --height=300
Bấm Q: Thoát

Cách 2: Training = prototxt
open cho 2 file 
+ model.caffemodel 
+ config.prototxt
+ https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
+ https://raw.githubusercontent.com/dkurt/cvpr2019/master/opencv_face_detector.prototxt
Lệnh cmd : python object_detection.py --model=res10_300x300_ssd_iter_140000.caffemodel --config=opencv_face_detector.prototxt --width=300 --height=300
Sửa dữ liệu mean

Cách 3: Training = object 
Link down model: https://pjreddie.com/media/files/yolov3.weights 
Link down config: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
Link down pjreddie data dog, horse: https://github.com/pjreddie/darknet/tree/master/data
Link tài liệu đọc dog, horse: https://pjreddie.com/darknet/yolo/
Lấy file object_detection_classes_pascal_voc.txt: http://bggit.ihub.org.cn/p30172569/opencv/blob/ff1ec6ccc6edb7dce92a9511be00c1bc62b01326/samples/data/dnn/object_detection_classes_pascal_voc.txt
Sửa rdb = true 
Sửa args.mean = [0, 0, 0]
'''

import cv2 as cv
import argparse
import numpy as np
import sys
import time
from threading import Thread
if sys.version_info[0] == 2:
    import Queue as queue
else:
    import queue

from common import * 
from tf_text_graph_common import readTextMessage
from tf_text_graph_ssd import createSSDGraph
from tf_text_graph_faster_rcnn import createFasterRCNNGraph


def main(nametype):
    global process, framesQueue, args, processedFramesQueue, predictionsQueue
    backends = (cv.dnn.DNN_BACKEND_DEFAULT, cv.dnn.DNN_BACKEND_HALIDE, cv.dnn.DNN_BACKEND_INFERENCE_ENGINE, cv.dnn.DNN_BACKEND_OPENCV,
                cv.dnn.DNN_BACKEND_VKCOM, cv.dnn.DNN_BACKEND_CUDA)
    targets = (cv.dnn.DNN_TARGET_CPU, cv.dnn.DNN_TARGET_OPENCL, cv.dnn.DNN_TARGET_OPENCL_FP16, cv.dnn.DNN_TARGET_MYRIAD, cv.dnn.DNN_TARGET_HDDL,
            cv.dnn.DNN_TARGET_VULKAN, cv.dnn.DNN_TARGET_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16)

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--zoo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models.yml'),
                        help='An optional path to file with preprocessing parameters.')
    #parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--out_tf_graph', default='graph.pbtxt',
                        help='For models from TensorFlow Object Detection API, you may '
                            'pass a .config file which was used for training through --config '
                            'argument. This way an additional .pbtxt file with TensorFlow graph will be created.')
    parser.add_argument('--framework', choices=['caffe', 'tensorflow', 'torch', 'darknet', 'dldt'],
                        help='Optional name of an origin framework of the model. '
                            'Detect it automatically if it does not set.')
    parser.add_argument('--thr', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.4, help='Non-maximum suppression threshold')
    parser.add_argument('--backend', choices=backends, default=cv.dnn.DNN_BACKEND_DEFAULT, type=int,
                        help="Choose one of computation backends: "
                            "%d: automatically (by default), "
                            "%d: Halide language (http://halide-lang.org/), "
                            "%d: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
                            "%d: OpenCV implementation, "
                            "%d: VKCOM, "
                            "%d: CUDA" % backends)
    parser.add_argument('--target', choices=targets, default=cv.dnn.DNN_TARGET_CPU, type=int,
                        help='Choose one of target computation devices: '
                            '%d: CPU target (by default), '
                            '%d: OpenCL, '
                            '%d: OpenCL fp16 (half-float precision), '
                            '%d: NCS2 VPU, '
                            '%d: HDDL VPU, '
                            '%d: Vulkan, '
                            '%d: CUDA, '
                            '%d: CUDA fp16 (half-float preprocess)' % targets)
    parser.add_argument('--async', type=int, default=0,
                        dest='asyncN',
                        help='Number of asynchronous forwards at the same time. '
                            'Choose 0 for synchronous mode')

    #Thu Hương thiết lập thêm thông tin model, config, classes thay vì thiết lập lúc chạy cmd
    if nametype == 'face_detect_opencv_dnn_caffe':                    
        parser.add_argument('--model', default='D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/Model/Model_face_detect_opencv_dnn_caffe/res10_300x300_ssd_iter_140000.caffemodel')
        parser.add_argument('--config', default='D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/Model/Model_face_detect_opencv_dnn_caffe/opencv_face_detector.prototxt')
        parser.add_argument('--classes', default=None)
        parser.add_argument('--input', default=None, help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    elif nametype == 'face_detect_opencv_dnn': 
        parser.add_argument('--model', default='D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/Model/Model_face_detect_opencv_dnn/opencv_face_detector_uint8.pb')
        parser.add_argument('--config', default='D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/Model/Model_face_detect_opencv_dnn/opencv_face_detector.pbtxt')
        parser.add_argument('--classes', default=None)
        parser.add_argument('--input', default=None, help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    elif nametype == 'object_detect_opencv_yolo3': 
        parser.add_argument('--model', default='D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/Model/Model_object_detect_yolo3/yolov3.weights')
        parser.add_argument('--config', default='D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/Model/Model_object_detect_yolo3/yolov3.cfg')
        parser.add_argument('--classes', default='D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/Model/Model_object_detect_yolo3/object_detection_classes_pascal_voc.txt')
        parser.add_argument('--input', default='D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/HinhAnh/Model_object_detect_yolo3/dog.jpg', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    args, _ = parser.parse_known_args()
    add_preproc_args(args.zoo, parser, 'object_detection')
    parser = argparse.ArgumentParser(parents=[parser],
                                    description='Use this script to run object detection deep learning networks using OpenCV.',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()

    args.model = findFile(args.model)
    args.config = findFile(args.config)
    args.classes = findFile(args.classes)
    args.input = findFile(args.input)

    #Thu Hương - Thiết lập thêm các thông số cho tùy loại name_type
    if nametype == 'object_detect_opencv_yolo3': 
        args.width = 416
        args.height = 416
        args.rgb = True
        args.sample = "object_detection"
        args.mean = [0, 0, 0]
        args.scale = 0.00392
        args.input = 'D:/LV/BTThuHuong_HCMUTE/XuLyAnh/DoAnThuHuong/HinhAnh/Model_object_detect_yolo3/dog.jpg'
    else : 
        args.width = 300
        args.height = 300
        args.rgb = False
        args.sample = "object_detection"
        args.mean = [104, 177, 123]
        args.scale = 1.0

    # If config specified, try to load it as TensorFlow Object Detection API's pipeline.
    config = readTextMessage(args.config)
    if 'model' in config:
        print('TensorFlow Object Detection API config detected')
        if 'ssd' in config['model'][0]:
            print('Preparing text graph representation for SSD model: ' + args.out_tf_graph)
            createSSDGraph(args.model, args.config, args.out_tf_graph)
            args.config = args.out_tf_graph
        elif 'faster_rcnn' in config['model'][0]:
            print('Preparing text graph representation for Faster-RCNN model: ' + args.out_tf_graph)
            createFasterRCNNGraph(args.model, args.config, args.out_tf_graph)
            args.config = args.out_tf_graph


    # Load names of classes
    classes = None
    if args.classes:
        with open(args.classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')

    # Load a network
    net = cv.dnn.readNet(cv.samples.findFile(args.model), cv.samples.findFile(args.config), args.framework)
    net.setPreferableBackend(args.backend)
    net.setPreferableTarget(args.target)
    outNames = net.getUnconnectedOutLayersNames()

    confThreshold = args.thr
    nmsThreshold = args.nms

    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        def drawPred(classId, conf, left, top, right, bottom):
            # Draw a bounding box.
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

            label = '%.2f' % conf

            # Print a label of class.
            if classes:
                assert(classId < len(classes))
                label = '%s: %s' % (classes[classId], label)

            labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])
            cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        layerNames = net.getLayerNames()
        lastLayerId = net.getLayerId(layerNames[-1])
        lastLayer = net.getLayer(lastLayerId)

        classIds = []
        confidences = []
        boxes = []
        if lastLayer.type == 'DetectionOutput':
            # Network produces output blob with a shape 1x1xNx7 where N is a number of
            # detections and an every detection is a vector of values
            # [batchId, classId, confidence, left, top, right, bottom]
            for out in outs:
                for detection in out[0, 0]:
                    confidence = detection[2]
                    if confidence > confThreshold:
                        left = int(detection[3])
                        top = int(detection[4])
                        right = int(detection[5])
                        bottom = int(detection[6])
                        width = right - left + 1
                        height = bottom - top + 1
                        if width <= 2 or height <= 2:
                            left = int(detection[3] * frameWidth)
                            top = int(detection[4] * frameHeight)
                            right = int(detection[5] * frameWidth)
                            bottom = int(detection[6] * frameHeight)
                            width = right - left + 1
                            height = bottom - top + 1
                        classIds.append(int(detection[1]) - 1)  # Skip background label
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        elif lastLayer.type == 'Region':
            # Network produces output blob with a shape NxC where N is a number of
            # detected objects and C is a number of classes + 4 where the first 4
            # numbers are [center_x, center_y, width, height]
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])
        else:
            print('Unknown output layer type: ' + lastLayer.type)
            exit()

        # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
        # or NMS is required if number of outputs > 1
        if len(outNames) > 1 or lastLayer.type == 'Region' and args.backend != cv.dnn.DNN_BACKEND_OPENCV:
            indices = []
            classIds = np.array(classIds)
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            unique_classes = set(classIds)
            for cl in unique_classes:
                class_indices = np.where(classIds == cl)[0]
                conf = confidences[class_indices]
                box  = boxes[class_indices].tolist()
                #ThuHuong sửa code 
                nms_indices = nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
                if nametype == 'face_detect_opencv_dnn_caffe' or nametype == 'face_detect_opencv_dnn':
                    nms_indices = nms_indices[:, 0] if len(nms_indices) else []
                else : nms_indices = nms_indices[:] if len(nms_indices) else []
                indices.extend(class_indices[nms_indices])
        else:
            indices = np.arange(0, len(classIds))

        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    # Process inputs
    winName = 'Deep learning object detection in OpenCV'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    def callback(pos):
        global confThreshold
        confThreshold = pos / 100.0

    cv.createTrackbar('Confidence threshold, %', winName, int(confThreshold * 100), 99, callback)

    cap = cv.VideoCapture(cv.samples.findFileOrKeep(args.input) if args.input else 0)

    class QueueFPS(queue.Queue):
        def __init__(self):
            queue.Queue.__init__(self)
            self.startTime = 0
            self.counter = 0

        def put(self, v):
            queue.Queue.put(self, v)
            self.counter += 1
            if self.counter == 1:
                self.startTime = time.time()

        def getFPS(self):
            return self.counter / (time.time() - self.startTime)


    process = True

    #
    # Frames capturing thread
    #
    framesQueue = QueueFPS()
    def framesThreadBody():
        global framesQueue, process

        while process:
            hasFrame, frame = cap.read()
            if not hasFrame:
                break
            framesQueue.put(frame)


    #
    # Frames processing thread
    #
    processedFramesQueue = queue.Queue()
    predictionsQueue = QueueFPS()
    def processingThreadBody():
        global processedFramesQueue, predictionsQueue, args, process

        futureOutputs = []
        while process:
            # Get a next frame
            frame = None
            try:
                frame = framesQueue.get_nowait()

                if args.asyncN:
                    if len(futureOutputs) == args.asyncN:
                        frame = None  # Skip the frame
                else:
                    framesQueue.queue.clear()  # Skip the rest of frames
            except queue.Empty:
                pass


            if not frame is None:
                frameHeight = frame.shape[0]
                frameWidth = frame.shape[1]

                # Create a 4D blob from a frame.
                inpWidth = args.width if args.width else frameWidth
                inpHeight = args.height if args.height else frameHeight
                blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=args.rgb, ddepth=cv.CV_8U)
                processedFramesQueue.put(frame)

                # Run a model
                net.setInput(blob, scalefactor=args.scale, mean=args.mean)
                if net.getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                    frame = cv.resize(frame, (inpWidth, inpHeight))
                    net.setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

                if args.asyncN:
                    futureOutputs.append(net.forwardAsync())
                else:
                    outs = net.forward(outNames)
                    predictionsQueue.put(np.copy(outs))

            while futureOutputs and futureOutputs[0].wait_for(0):
                out = futureOutputs[0].get()
                predictionsQueue.put(np.copy([out]))

                del futureOutputs[0]


    framesThread = Thread(target=framesThreadBody)
    framesThread.start()

    processingThread = Thread(target=processingThreadBody)
    processingThread.start()

    #
    # Postprocessing and rendering loop
    #
    while cv.waitKey(1) < 0:
        try:
            # Request prediction first because they put after frames
            outs = predictionsQueue.get_nowait()
            frame = processedFramesQueue.get_nowait()

            postprocess(frame, outs)

            # Put efficiency information.
            if predictionsQueue.counter > 1:
                label = 'Camera: %.2f FPS' % (framesQueue.getFPS())
                cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                label = 'Network: %.2f FPS' % (predictionsQueue.getFPS())
                cv.putText(frame, label, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

                label = 'Skipped frames: %d' % (framesQueue.counter - predictionsQueue.counter)
                cv.putText(frame, label, (0, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

            cv.imshow(winName, frame)
        except queue.Empty:
            pass
        
    process = False
    framesThread.join()
    processingThread.join()
    
if __name__ == '__main__':
    #Gọi truyền biến tương ứng cho từng loại model 
    #main('face_detect_opencv_dnn_caffe')
    #main('face_detect_opencv_dnn')
    main('object_detect_opencv_yolo3')
