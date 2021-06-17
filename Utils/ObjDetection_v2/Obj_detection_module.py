import cv2

class ObjectDetection:
    def __init__(self):

        self.thres = 0.45 # Threshold to detect object
    def detect(self,obj_img,show=False,draw=False):
        img = cv2.imread(obj_img)

        classNames= []
        classFile = 'coco.names'
        with open(classFile,'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightsPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)


            
        classIds, confs, bbox = net.detect(img,confThreshold=self.thres)
        #print(classIds,bbox)
        Object_names = []
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                Object_names.append(classNames[classId-1])
                if draw:
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        if show:
            cv2.imshow("Output",img)
            cv2.waitKey(0) 
            return Object_names
        else:
            return Object_names            

if __name__ == '__main__':
    obj = ObjectDetection()
    obj_img = "apple.jpg"
    print(obj.detect(obj_img,show=True,draw=False))