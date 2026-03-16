
import serial
import time
import xlsxwriter
import cv2
import os
import random 
#16px = 1mm -> 1px = 0,0625

#Communication setup:
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=640,
    display_width=640,
    display_height=640,
    framerate=1,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


arduino = serial.Serial('/dev/ttyACM0', 115200, timeout = 1)
time.sleep(2)
cam = cv2.VideoCapture(1)
im_count = 0
x, y, z, r = 0, 0, 0, 0
folder = "map_move_2"

if folder not in os.listdir("./"):
    os.mkdir(folder)

#Creating workbook

workbook = xlsxwriter.Workbook(f'{folder}/Dataset.xlsx') 
worksheet = workbook.add_worksheet()
row = 0

#Data sent:
t = time.time()


def pic_taking():
    global im_count, t

    for r in range(int((time.time() - t) * 1) + 5):
        _, frame = cam.read()
    
    print("Image took")
    x, y = frame.shape[0] // 2, frame.shape[1] // 2
    size = min(x, y)
    t = time.time()
    cv2.imwrite(f'{folder}/img{im_count}.png', frame[x - size : x + size, y - size : y + size])
    im_count += 1



def send(m, x,y,z,r):
    move = f"M{m} X{x} Y{y} Z{z} R{r}\n"
    arduino.write(move.encode())
    print("Coordinates sent", move.strip())

#Data received:

def receive():
    global row, x, y, z, r
    if arduino.in_waiting > 0:
        answer = arduino.readline().decode().strip()
        print("Movement has been: ", answer)

        # Writing data to a column manually
        col = 0
        for item in [x, y, z, r]:
            worksheet.write(row, col, item)
            col += 1
        row += 1

        pic_taking()

        return answer
    return None
def rand_list():
    li = []
    xr = random.randint(-5,5)
    yr = random.randint(-5,5)
    zr = random.randint(-6,0)
    rr = random.randint(-30,30)
    li = [xr,yr,zr,rr]
    return li

l = rand_list()
final_list = []

for i in range(30):
    x_inc = random.choice([-1,-2,4,5])#change the increments here
    y_inc = random.choice([-3,-4])# change the increments here
    middle_list = [x_inc, y_inc,random.randint(-6,0),random.randint(-30,30)]
    
    l[0]= min(max(l[0]+middle_list[0],-180),180)
    l[1]= min(max(l[1]+middle_list[1],-90),90)
    l[2]= max(middle_list[2],-6)
    l[3]= (middle_list[3]+l[3])%30

    final_list.append(list(l))
final_list.append([0,0,0,0])


#Main loop:
#z max = 0
#z min = -6

#y max = 80
#y min = -70

#ydeltamax = 10
#xdeltamax = 25
#zdeltamax = 7

c_test = [
    #pic 1
    (0,0,-1,0),
    (0,0,-5,0),
    (0,0,-6,0),
    (0,0,0,0)
    #(0,0,-5,0),
    #(0,0,-6,0),
    #(0,0,-7,0),
    #(0,0,-8,0)
    ]

c_move = [
    #test (15 iterations)
    (0,0,-3,0),
    (5,-5,-3,0),
    (10,0,-3,0),
    (15,5,-3,0),
    (20,10,-3,0),
    (25,5,-3,0),
    (30,10,-3,0),
    (35,15,-3,0),
    (30,20,-3,0),
    (25,25,-3,0),
    (20,20,-3,0),
    (15,25,-3,0),
    (10,20,-3,0),
    (10,15,-3,0),
    (5,10,-3,0),
    (1,5,-3,0),
    #back to the origin
    (0,0,0,0)
]

c_move_rotate = [
    #test (20 iterations)
    (0,0,-35,0),
    (-10,-5,-35,0),
    (20,-10,-35,15),
    (40,-5,-35,30),
    (70,0,-35,45),
    (90,5,-35,60),
    (120,10,-35,75),
    (140,5,-35,90),
    (150,0,-35,105),
    (150,-5,-35,120),
    (160,-10,-35,60),
    (170,-5,-35,210),
    (160,0,-35,180),
    (145,5,-35,260),
    (120,10,-35,330),
    (95,15,-35,0),
    (70,20,-35,90),
    (45,15,-35,180),
    (20,10,-35,270),
    (10,5,-35,0),
    #back to the origin
    (0,0,0,0)
]

c_move_scale = [
    
    (-25,10,-5,0),
    (-35,20,-15,0),
    (-45,30,-15,0),
    (-30,40,-30,0),
    (-10,50,-35,0),
    (-5,60,-35,0),
    (5,55,-35,0),
    (20,50,-35,0),
    (25,40,-35,0),
    (35,30,-35,0),
    (20,20,-25,0),
    (40,10,0,0),            
    (30,0,-20,0),
    (20,-10,-10,0),
    (20,-20,-15,0),
    (10,-15,-25,0),
    (10,-10,-30,0),
    #Back to origin
    (0,0,0,0)
                  
                
     ]

c_blured = []


c_all1 = []

c_all2 = []

c_all3 = []

c_all4 = []

c_all5 = []

c_all6 = []
 
#x, y, z, r
commands = final_list


try:
    send(2, 0, 0, 0, 0)
    pic_taking()
    for command in commands:
        receive()
        
        x = command[0]
        y = command[1] 
        z = command[2]
        r = command[3]

#Call previously defined functions:

        send(0, x,y,z,r)


        while arduino.in_waiting == 0:
            pass #wait until arduino arrives

        time.sleep(1.5)
        receive()
    workbook.close()
    

#Exception (required for try) and closure if disconnected

except KeyboardInterrupt:
    send(1, 0, 0, 0, 0)

    print("\n")
    print("Program has been stopped.")

finally:
    arduino.close()
    workbook.close()
    print("Communication Jetson - Arduino stopped")