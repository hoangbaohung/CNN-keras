import time
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
from PIL import Image
from PIL import ImageFilter
from PIL.ImageFilter import GaussianBlur
from io import BytesIO
import requests
import thread


imgSize = (1920,1080)
camera = PiCamera()
camera.resolution = imgSize
camera.shutter_speed = 12000
awb_mode = 'auto'
rawCapture = PiRGBArray(camera, size=imgSize)
captureInterval = 0.3
global moving_threshold
global activeRequest
activeRequest = 0
moving_threshold = 500
def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def captureAndUploadImage():
    global activeRequest
    stream = BytesIO()
    activeRequest += 1 
    camera.capture("{0}.jpg".format(imgCount),format='jpeg', resize=(1920,1080))
    camera.capture(stream, format='jpeg')
    stream.seek(0)
    url = 'http://192.168.1.216:9981/upload'
    try:
        time.sleep(1)
        response = requests.post(url,files = {'image': stream.read()})
        print("upload ok{0}".format(time.time()))
    except Exception:
        print (Exception)
    activeRequest -= 1 
    stream.truncate()

def update_threshold():
    global moving_threshold
    url = 'http://192.168.1.216:9981/update_threshold'
    while True:
        try:
            response = requests.get(url)
            print(response.text)
            moving_threshold = int(response.text)
            time.sleep(30)
        except: pass

    
time.sleep(1)
lastImage = None
count=0
#thread.start_new_thread(update_threshold,())
for output in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    count +=1
    image = output.array 
    if lastImage == None:
        lastImage = image
        rawCapture.truncate(0)
        continue
    image = Image.fromarray(image)
    lastImage = Image.fromarray(lastImage)
    image = image.convert('L')
    lastImage = lastImage.convert('L')
    image = image.filter(ImageFilter.BLUR)
    lastImage = lastImage.filter(ImageFilter.BLUR)
    image = np.array(image)
    lastImage = np.array(lastImage)
    delta = np.sum(abs(weightedAverage(image)-weightedAverage(lastImage) > 10))
    print(delta)
    print (moving_threshold)
    if delta > moving_threshold:
        #print(activeRequest)
        if activeRequest > 2:
            print("wating network queue")
            rawCapture.truncate(0)
            continue
        try:
            thread.start_new_thread(captureAndUploadImage,())
        except: pass
    rawCapture.truncate(0)
    time.sleep(captureInterval)
