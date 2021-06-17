import cv2
import numpy as np

class ObjectDetection:
    def __init__(self):
        # Low Value for more effect
        self.thres = 0.45 # Threshold to detect object
        self.nms_threshold = 0.2
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(3,1280)
        # self.cap.set(4,720)
        # self.cap.set(10,70)

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

        indices = cv2.dnn.NMSBoxes(bbox,confs,self.thres,self.nms_threshold) # Non maximum Supression
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

    def Obj_detection_video(self,show=True,draw=True):
        output = list()
        while True:
            success,img = self.cap.read()
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

            indices = cv2.dnn.NMSBoxes(bbox,confs,self.thres,self.nms_threshold) # Non maximum Supression
            #print(indices)

            Obj_names = []
            for i in indices:
                i = i[0]
                Obj_names.append(classNames[classIds[i][0]-1])
                if draw:
                    box = bbox[i]
                    x,y,w,h = box[0],box[1],box[2],box[3]
                    rec = False
                    if rec:
                        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
                    cv2.putText(img,classNames[classIds[i][0]-1].upper(),(box[0]+10,box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    output.append(classNames[classIds[i][0]-1].upper())                
            if show:
                cv2.imshow("Output",img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    output = set(output)
                    return output

    def obj_detection_v2(self):
        classNames= []
        classFile = 'Utils/ObjDetection_v2/coco.names'
        with open(classFile,'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'Utils/ObjDetection_v2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'Utils/ObjDetection_v2/frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightsPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        output = list()
        while True:
            success,img = self.cap.read()
            classIds, confs, bbox = net.detect(img,confThreshold=self.thres)
            # print(classIds,bbox)

            if len(classIds) != 0:
                for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                    recbox = False
                    if recbox == True:
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    output.append(classNames[classId-1].upper())            
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            cv2.imshow("Output",img)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                output = set(output)
                self.cap.release()
                cv2.destroyAllWindows()
                return output                
        


if __name__ == "__main__":
    
    obj = ObjectDetection()
    print(obj.obj_detection_v2())            