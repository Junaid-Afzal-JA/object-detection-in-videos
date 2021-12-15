import os
import csv
import cv2
import numpy as np
from tkinter.filedialog import askdirectory
from tkinter import filedialog

# image_name = filedialog.askopenfilename()
#
# image = cv2.imread(image_name)
# cv2.imshow('out[ut', image)
# cv2.waitKey(5)


class DetectObjectFromImage:
    """
    Detect Objects from Image
    """

    def __init__(self):
        """
        Message:
            Setting Up Object classes (coco.names) and Yolo
        """

        self.nmsThreshold = 0.2
        self.required_size = 320

        self.yolo = cv2.dnn.readNet('yolo_object_detection/yolov3.cfg', 'yolo_object_detection/yolov3.weights')
        # yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layer_names = self.yolo.getLayerNames()

        self.classes = []
        with open('yolo_object_detection/coco.names', 'r') as file:
            lines = file.readlines()
            self.classes = [line.rstrip() for line in lines]

    def find_objects(self, outputs, img, image_name):
        height, width, channel = img.shape
        bounding_box = []  # contains -> x, y, w, h
        class_ids = []  # contains all class IDS
        confs = []  # confidence value

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:

                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x, y = int((detection[0] * width) - w / 2), int((detection[1] * height) - h / 2)
                    bounding_box.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))

        # For taking care of overlapping boxes
        indices = cv2.dnn.NMSBoxes(bounding_box, confs, 0.5, self.nmsThreshold)
        # print(indices)
        objects_list = []
        for i in indices:
            # i = indices[0]
            box = bounding_box[i]
            # print(bounding_box)
            x, y, w, h = box[0], box[1], box[2], box[3]
            # cv2.rectangle(img, (x, y), conner points, color, thickness)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(img, f'{self.classes[class_ids[i]].upper()}{int(confs[i] * 100)}%',
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            # print("\n" + image_name + " is " + self.classes[class_ids[i]])
            objects_list.append(self.classes[class_ids[i]])
        return objects_list

    def detecting_objects_from_path(self, path):
        """
        Message:
            Read images from provided path
            and create detected-objects list.
        Parameters:
            path: Path of folder containing images.
        Return:
            None
        """
        images_list = os.listdir(path)
        detection_result = {}
        for image_name in images_list:
            image_path = path + '/' + image_name
            if os.path.isfile(image_path):
                image = cv2.imread(image_path)
                blob = cv2.dnn.blobFromImage(image, 1 / 255,
                                             (self.required_size, self.required_size),
                                             [0, 0, 0], 1, crop=False
                                             )

                self.yolo.setInput(blob)
                layers_names = self.yolo.getLayerNames()
                # print(type(layersNames))
                # for i in yolo.getUnconnectedOutLayers():
                #     print(i)
                output_names = [layers_names[i - 1] for i in self.yolo.getUnconnectedOutLayers()]

                outputs = self.yolo.forward(output_names)
                detection_result[image_name] = self.find_objects(outputs, image, image_name)

        self.convert_to_csv(detection_result)
        return detection_result

    def convert_to_csv(self, data):
        """
        Convert the dictionary to csv file:
        Parameters:
            data (dict):
        """
        p = askdirectory()
        with open(p + '/dct.csv', 'w') as f:
            writer = csv.writer(f)
            for k, v in data.items():
                writer.writerow([k, v])

# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyWindow('Image')


# dec = DetectObjectFromImage()
#
# result = dec.detecting_objects_from_path('images_data')
# # print(result)
# for key in result:
#     print(f'{key}:\t {result[key]}')
