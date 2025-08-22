// CO2 Detection with CSV Logging
const int sensorPin = A0;  
int sensorValue = 0;       
float voltage = 0;
float co2_ppm = 0;

void setup() {
  Serial.begin(9600);
  Serial.println("Time (s),Raw Value,Voltage (V),CO2 (ppm)");  // CSV Header
}

void loop() {
  static unsigned long startTime = millis();  // Start time
  unsigned long elapsedTime = (millis() - startTime) / 1000;  // seconds

  sensorValue = analogRead(sensorPin);
  voltage = sensorValue * (5.0 / 1023.0);
  co2_ppm = sensorValue * (5000.0 / 1023.0);

  // Print in CSV format
  Serial.print(elapsedTime);
  Serial.print(",");
  Serial.print(sensorValue);
  Serial.print(",");
  Serial.print(voltage, 3);  // 3 decimal places
  Serial.print(",");
  Serial.println(co2_ppm, 2); // 2 decimal places

  delay(1000);
}
