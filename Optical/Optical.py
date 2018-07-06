import time,  cv2,  os,  win32file
import matplotlib.pyplot as plt
#import plotly.plotly as py
import numpy as np
import datetime

## ZED ##
# Docuemntation: https://www.stereolabs.com/developers/documentation/API/v2.4.0/
import pyzed.camera as zcam
import pyzed.types as tp
import pyzed.core as core
import pyzed.defines as sl


runtime = zcam.PyRuntimeParameters()
zedImg = core.PyMat()
####

#######################      CLASSES AND FUNCTIONS      #######################

def loadImage(myCam):
    # twoImages = cv2.imread("D:/data/5/img_000000010.ppm")
    #img = myCam.read()
    err = myCam.grab(runtime)
    myCam.retrieve_image(zedImg)
    return zedImg.get_data()
    
    
def timings(list):
    print("Timing-List")
    for foo in range(len(list)-1):
        print(str(round(list[foo+1][1]-list[foo][1], 6))+ ": " + str(list[foo+1][0]))


def initCam():
    #myCam = cv2.VideoCapture(0) # Other camera than ZED
    myCam = zcam.PyZEDCamera()  # only for ZED camera
    if not myCam.is_opened():
        print("Opening ZED Camera...")
    init = zcam.PyInitParameters()
    status = myCam.open(init)
    if status != tp.PyERROR_CODE.PySUCCESS:
        print(repr(status))
        exit()
    return myCam



#######################             INIT                #######################
print ("\n################  Testbahn Tool  #####################")
cv2.destroyAllWindows()
totalframes   = 0
startedFrames = 0
hdd = win32file.GetDiskFreeSpace("C:/")
print("Freier Speicherplatz: %s GB" %(round(hdd[0]*hdd[1]*hdd[2]/1024/1024/1024, 2)))
start = time.time()




#TestID & Unterordner erstellen
testID = input("Bitte TestID eingeben: ")
testIDRound = 0
folder = str(testID) + "/" + str(testIDRound) 
while(os.path.exists(folder)): 
    testIDRound +=1
    folder = str(testID) + "/" + str(testIDRound) 
os.makedirs(folder)


#create Camera
cap = initCam()
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#cap.set( , )  Helligkeit



print ("\n########   Testbild wird angezeigt   ############\nESC um Messung zu starten")
while(1):
    img = loadImage(cap)
    cv2.imshow("image", img)
    k = cv2.waitKey(5)
    if k == 27: 
        break  # esc to quit
cv2.destroyAllWindows()

#Zusätzliche Infos als txt
f = open(folder+"/data.txt", "w")
now = datetime.datetime.now()
datestr = "%02d.%02d.%04d / %02d:%02d.%02d" % (now.day,  now.month,  now.year,  now.hour,  now.minute,  now.second)
f.write("Start der Messung: " + datestr + "\n")
f.close()

##create mask for exposurecontroll
#x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#myMask = np.zeros((y,x),  dtype = np.uint8)
#points = np.array([[400, 0], [x-400, 0], [x-200, y],  [200, y]], np.int32)
#cv2.fillConvexPoly(myMask, points , 255)


start = time.time()
print ("\n########   Aufzeichnung gestartet!   ############")
print ("Abbruch mit Strg + C ")
time.sleep(0.5)

filepath = folder+"/vid.avi"
vid = cap.enable_recording(filepath)
print("Recording started...")
while(1): 
    try:
        startedFrames +=1
        #now = datetime.datetime.now()
        #datestr = "%02d.%02d.%04d / %02d:%02d.%02d" % (now.day,  now.month,  now.year,  now.hour,  now.minute,  now.second)

        print ("\n_____________________  Frame %s  _____________________" % startedFrames)

        #Load Images
        image = loadImage(cap)
        if image is None: 
            print("Missing Frame") 
            continue
        cv2.imshow("frame",  image)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        #cv2.waitKey()
        #splitted = cv2.split(image)
        #hist0 = cv2.calcHist(splitted[0], [0], None , [256] , [0, 255])
        #plt.plot(hist0)   
        #plt.close()  
        #plt.show(block = False)
        #mean = cv2.mean(image, myMask)
        #print(mean)

        

        cap.record()

        #time.sleep(0.5)
    except KeyboardInterrupt: 
        print("\n!!!      Speicherung beendet      !!!")
        break
    
    
cap.disable_recording()
print("Recording finished.")
end = time.time()
now = datetime.datetime.now()
datestr = "%02d.%02d.%04d / %02d:%02d.%02d" % (now.day,  now.month,  now.year,  now.hour,  now.minute,  now.second)
f = open(folder+"/data.txt", "a")
f.write("Ende der Messung:  " + datestr + "\nAnzahl Frames: %s\nT = %s Sekunden\n%s FPS" % (totalframes,  round((end-start), 3),  round((totalframes/(end-start)), 3)))
f.close()



#timings(timelist)
print ("Anzahl Frames: %s \nT = %s Sekunden, %s FPS " % (totalframes,  round((end-start), 3),  round((totalframes/(end-start)), 3)))

  
