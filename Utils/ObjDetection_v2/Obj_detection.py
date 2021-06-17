import cv2

thres = 0.45 # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

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


def obj_detection(output):

    while True:
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=thres)
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
            return output

if __name__=='__main__':
    
    print(obj_detection())            