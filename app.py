import serial
import csv
import time

# Connect to Arduino on COM5
ser = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)  # give time for Arduino to reset

# Start timer
start_time = time.time()

# Open CSV file for writing
with open("co2_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time (s)", "Raw Value", "Voltage (V)", "CO2 (ppm)"])  # header row

    print("Logging data from Arduino on COM5... Press CTRL+C to stop.")

    try:
        while True:
            line = ser.readline().decode(errors='ignore').strip()  # read line from Arduino
            if line and "," in line:
                data = line.split(",")  # e.g. ["512", "2.5", "400"]

                # Add elapsed time
                elapsed = round(time.time() - start_time, 2)
                row = [elapsed] + data

                writer.writerow(row)
                f.flush()  # force write to file
                print(row)  # show live on console

    except KeyboardInterrupt:
        print("\nLogging stopped. File saved as co2_data.csv")