import cv2
import pytesseract

# Tesseract ocr Executable file location
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

class TextDetection:

    def img_load(self,txt_img):
        img = cv2.imread(txt_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    # Image to String
    # print(pytesseract.image_to_string(img))

    # Detecting Character

    def detect_chr(self,txt_img,draw=True,show=True):
        img = self.img_load(txt_img)
        hImg, wImg, _ = img.shape
        output_txt = []
        boxes = pytesseract.image_to_boxes(img)
        for b in boxes.splitlines():
            #print(b)
            b = b.split(' ')
            #print(b)
            output_txt.append(b[0])
            
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            if draw:
                cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 2)
            cv2.putText(img, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        if show:
            cv2.imshow('Result', img)
            cv2.waitKey(0)
            return output_txt
        else:
            return output_txt    


    # Detecting Words

    def detect_words(self,txt_img,draw=True,show=True):
        img = self.img_load(txt_img)
        hImg, wImg, _ = img.shape
        output_txt = []
        boxes = pytesseract.image_to_data(img)
        for a, b in enumerate(boxes.splitlines()):
            if a != 0:
                b = b.split()
                #print(b)
                
                if len(b) == 12:
                    output_txt.append(b[11])
                    if draw:
                        x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 255), 2)
                        cv2.putText(img, b[11], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
        if show:
            cv2.imshow('Result', img)
            cv2.waitKey(0)
            return output_txt
        else:
            return output_txt    


    # Detecting ONLY Digits

    def detect_digits(self,txt_img,draw=True,show=True):
        img = self.img_load(txt_img)
        hImg, wImg, _ = img.shape
        output_txt = []
        conf = r'--oem 3 --psm 6 outputbase digits'
        boxes = pytesseract.image_to_boxes(img, config=conf)
        for b in boxes.splitlines():
            #print(b)
            b = b.split(' ')
            #print(b)
            output_txt.append(b[0])
            if draw:
                x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 2)
                cv2.putText(img, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

        if show:        
            cv2.imshow("Result", img)
            cv2.waitKey(0)
            return output_txt
        else:
            return output_txt    


    # Webcam and Screen Capture
    def detect_from_webcam(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        output_txt = []
        draw = False
        while True:
            timer = cv2.getTickCount()
            _, img = cap.read()
            # DETECTING CHARACTERS
            hImg, wImg, _ = img.shape
            boxes = pytesseract.image_to_boxes(img)
            
            for b in boxes.splitlines():
                # print(b)
                b = b.split(' ')
                # print(b)
                output_txt.append(b[0])
                x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
                if draw:
                    cv2.rectangle(img, (x, hImg - y), (w, hImg - h), (50, 50, 255), 2)
                cv2.putText(img, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
            #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            #cv2.putText(img, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 230, 20), 2)
            cv2.imshow("Result", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return output_txt


if __name__ == '__main__':

    choice = input("Enter your choice : \n 1.detect character \n 2.detect words \n 3.detect digits \n 4.detect "
                   "character using webcam\n")
    obj = TextDetection()    
    txt_img = 'img/test2.jpg'           
    if choice == "1":
        print(obj.detect_chr(txt_img))
    elif choice == "2":
        print(obj.detect_words(txt_img))
    elif choice == "3":
        print(obj.detect_digits(txt_img))
    elif choice == "4":
        print(obj.detect_from_webcam())
    else:
        print("Invalid Choice")
