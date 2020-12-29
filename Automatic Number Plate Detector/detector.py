import cv2
import numpy as np
import pytesseract    # OCR -> Optical character recognition engine  used to find text from any image

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
cascade=cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
states = {
       "AN":"Andaman and Nicobar","AP":"Andhra Pradesh","AR":"Arunachal Pradesh","AS":"Assam",
       "BR":"Bihar","CH":"Chandigarh","CG":"Chattisgarh","DD":"Dadra and Nagar Haveli and Daman and Diu",
       "DL":"Delhi","GA":"Goa","GJ":"Gujrat","HR":"Haryana","HP":"Himachal Pradesh","JK":"Jammu and Kashmir",
       "JH":"Jharkhand","KA":"Karnataka","KL":"Kerala","LA":"Ladakh","LD":"Lakshdweep","MP":"Madhya Pradesh",
       "MH":"Maharastra","MN":"Manipur","ML":"Megahlaya","MZ":"Mizoram","NL":"Nagaland","OD":"Odisha",
       "PY":"Puducherry","PB":"Punjab","RJ":"Rajasthan","SK":"Sikkim","TN":"Tamil Nadu","TS":"Telangana",
       "TR":"Tripura","UP":"Uttar Pradesh","UK":"Uttarakhand","WB":"West Bengal"
}

def ext_num(img_name):
    global read
    img=cv2.imread(img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in nplate:
        # Now we are cropping the number plate of the car from the image in next two line
        a,b = (int(0.02*img.shape[0]),int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a,x+b:x+w-b,:]

        # Now here we do some Image Processing
        kernel = np.ones((1,1),np.uint8)
        plate = cv2.dilate(plate,kernel,iterations=1)
        plate = cv2.erode(plate,kernel,iterations=1)
        plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh,plate)=cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Use of OCR

        read=pytesseract.image_to_string(plate) # read image and return the text fromat of content avalible in image
        read=''.join(e for e in read if e.isalnum())   # is alpha number
        state = read[0:2]
        try:
            print('Car Belongs to : ',states[state])
        except:
            print('State not recognised')
        print(read)
        cv2.rectangle(img, (x,y), (x+w, y+h), (51,51,255), 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y),(51,51,255) , -1)
        cv2.putText(img,read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Plate',plate)

    cv2.imshow('Result',img)
    cv2.imwrite('result.jpg', plate)   
    cv2.waitKey(0) 

ext_num('image1.jpg')




