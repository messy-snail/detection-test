import serial
import time
import json
 
msg = {'conveyor_step' : 0, 'sorting_step' : 0}
port = '/dev/ttyUSB0' # 시리얼 포트
baud = 9600 # 시리얼 보드레이트(통신속도)
 
ser = serial.Serial(port,baud)
 
end_str = '\n'
 
while True:
    print("conveyor_step_Input: ")
    msg['conveyor_step'] = input() 
    print("sorting_step_Input: ")
    msg['sorting_step'] = input() 
    json_msg = json.dumps(msg)
    ser.write(json_msg.encode())
    ser.write(end_str.encode())
