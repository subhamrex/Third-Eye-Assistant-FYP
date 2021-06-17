import pytesseract
import cv2
import re

import numpy as np

# Tesseract ocr Executable file location
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class detect_text:
    save_dir="Utils/TextDetection/image/"
    test_img_dir="Utils/TextDetection/image/test/"
    fileName=None
    image=None
    target=None
    url_pattern="http[s]?://(?:[a-zA-Z]|[0-9]|[.]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    email_pattern="([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
    phone_pattern="^([+]|[9][1]|[0])([0-9]{10})+"
    dst=None
    text_detection_flag=image_process_flag=0
    
    text=None


    def validate_image(self,img):
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        print(laplacian_var)
        if laplacian_var < 5:
            return False
        else:
            return True


    def mapper(self,h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
        return hnew


    def __init__(self,filename):
        self.fileName=filename
        self.save_dir=f"{self.save_dir}{self.fileName}"
    

    def image_preprocess(self):
        pts=np.float32([[0,0],[800,0],[800,800],[0,800]])

        img=self.image=cv2.resize(cv2.imread(self.save_dir),(800,800))

        if(self.validate_image(img) != True):
            print('Please take image again')
            return



        orig=self.image.copy()
        gray=cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(gray,(5,5),0)
        edge=cv2.Canny(blurred,30,50)
        contours,hierarchy=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours,key=cv2.contourArea,reverse=True)
        for c in contours:
            p=cv2.arcLength(c,True)
            approx=cv2.approxPolyDP(c,0.02*p,True)

            if (len(approx) == 4):
                self.target=approx
                break
        approx=self.mapper(self.target)
        op=cv2.getPerspectiveTransform(approx,pts)
        self.dst=cv2.warpPerspective(orig,op,(800,800))
        self.image_process_flag=1

        
        
    def show_image(self):
        if self.image_process_flag == 0 :self.image_preprocess()

        cv2.imshow("image",self.dst)
        cv2.waitKey(0)


    def detect_text(self):
        if self.image_process_flag == 0 :self.image_preprocess()
        self.text=pytesseract.image_to_string(self.dst)
        self.text=self.text.replace(','," ")
        self.text=self.text.replace('\n'," ")
        self.text=self.text.replace('\\'," ")
        self.text=self.text.replace('  '," ")
        self.text_detection_flag=1
        # self.text=self.text.split(" ")
        print(self.text)
    
    def fetch_info(self):
        if self.text_detection_flag == 0 : self.detect_text()
        urls = re.findall(self.url_pattern,self.text)
        emails = re.findall(self.email_pattern, self.text)
        phoneNums=re.findall(self.phone_pattern,self.text)

        print(urls,'\n',emails,'\n',phoneNums,'\n')


    def test(self):
        print(self.save_dir)



dt=detect_text("no029.jpg")

# dt.image_preprocess()
dt.show_image()
dt.detect_text()