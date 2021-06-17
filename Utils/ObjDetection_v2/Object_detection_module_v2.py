import cv2
import numpy as np

class ObjectDetection:
    def __init__(self):
        self.thres = 0.45 # Threshold to detect object
        self.nms_threshold = 0.2

    def detect(self,obj_img,show=True,draw=True):
        img = cv2.imread(obj_img)
        classNames= []
        classFile = 'Utils/ObjDetection_v2/coco.names'
        with open(classFile,'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        #print(classNames)
        configPath = 'Utils/ObjDetection_v2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'Utils/ObjDetection_v2/frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightsPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

      
            
        classIds, confs, bbox = net.detect(img,confThreshold=self.thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))
        #print(type(confs[0]))
        #print(confs)

        indices = cv2.dnn.NMSBoxes(bbox,confs,self.thres,self.nms_threshold)
        #print(indices)

        Obj_names = []
        for i in indices:
            i = i[0]
            Obj_names.append(classNames[classIds[i][0]-1])
            if draw:
                box = bbox[i]
                x,y,w,h = box[0],box[1],box[2],box[3]
                cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
                cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        if show:
            cv2.imshow("Output",img)
            cv2.waitKey(0)
            return Obj_names
        else:
           return Obj_names     

if __name__ == "__main__":
    obj = ObjectDetection()
    obj_img = "Utils/ObjDetection_v2/apple.jpg"
    print(obj.detect(obj_img,show=False))            