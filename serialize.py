import serial
import time

# Connect to Arduino Uno
ser = serial.Serial('COM3', 9600)  # Replace 'COMX' with the appropriate serial port

# Wait for Arduino to initialize
time.sleep(2)

file_path = r"C:\Users\Sanora Teressa Raju\Desktop\ASL_REALTIME\logs\sam.txt"

with open(file_path, 'r') as file:
    word=file.read().lower().strip()
    if(word == "stop"):
        gesture_first_char = 't'
    else:
        # Move the file cursor back to the beginning
        file.seek(0)
        gesture_first_char = file.read(1).lower()

# Send a command to Arduino
command = gesture_first_char  # Customize this command as per your needs
print("command is",command)
ser.write(command.encode())

# Close the serial connection
ser.close()
