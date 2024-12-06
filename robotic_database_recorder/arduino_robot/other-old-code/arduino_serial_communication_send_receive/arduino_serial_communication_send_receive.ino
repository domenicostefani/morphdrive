/**
* 
* This sketch reads multiple analog and digital sensors as well as the IMU, and sends the values on the serial port.
* This sketch also receives meessages from the serial port in the format [string,number], and then
* associates to this messages a certain behavior (e.g, triggering a vibration motor pattern)
* 
* The analog sensors values are filtered with a butterworth lowpass filter.
* The filtering is achieved by means of the library https://github.com/tttapa/Filters
* The coefficients for the filter are calculated using the tools: http://www.exstrom.com/journal/sigproc/
* 
* 
* Author: Luca Turchet
* Date: 30/05/2019
* 
* 
* 
* 
**/

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <EEPROM.h>
#include <IIRFilter.h>
#include <string.h>

#define BAUD_RATE 115200 //NOTE: on the Teensy this is meaningless as the Teensy always transmits data at the full USB speed


/* Variables for incoming messages *************************************************************/

const byte MAX_LENGTH_MESSAGE = 64;
char received_message[MAX_LENGTH_MESSAGE];

char START_MARKER = '[';
char END_MARKER = ']';

    
boolean new_message_received = false;




/* Digital inputs *************************************************************/

const uint16_t digital_input0_pin = 0;
const uint16_t digital_input1_pin = 1;
const uint16_t digital_input2_pin = 2;

uint16_t digital_input0 = 0;
uint16_t digital_input1 = 0;
uint16_t digital_input2 = 0;

int digital_input0_button_state;
int digital_input1_button_state;
int digital_input2_button_state;

int digital_input0_last_button_state = LOW;
int digital_input1_last_button_state = LOW;
int digital_input2_last_button_state = LOW;   



/* Digital outputs *************************************************************/

const uint16_t digital_output0_pin = 3; 
const uint16_t digital_output1_pin = 4; 
const uint16_t digital_output2_pin = 5; 
const uint16_t digital_output3_pin = 9; 
const uint16_t digital_output4_pin = 10; 

int digital_output0_LED_state = HIGH;  
int digital_output1_LED_state = HIGH;  
int digital_output2_LED_state = HIGH;    


unsigned long digital_input0_last_debounce_time = 0;  
unsigned long digital_input1_last_debounce_time = 0;  
unsigned long digital_input2_last_debounce_time = 0;  

unsigned long debounce_delay = 50;    // the debounce time; increase if the output flickers







/* Analog inputs ******************************************************************************************/

//#define ANALOG_BIT_RESOLUTION 12 // Only for Teensy

//static const unsigned long ANALOG_PERIOD_MILLISECS = 1; // E.g. 4 milliseconds per sample for 250 Hz
//static const unsigned long ANALOG_ANALOG_PERIOD_MICROSECS = 1000 * PERIOD_MILLISECS;
//static const float ANALOG_SAMPLING_FREQUENCY = 1.0e3f / PERIOD_MILLISECS;
#define ANALOG_PERIOD_MICROSECS 1000
static uint32_t analog_last_read = 0;


uint16_t analog_input0_pin = 0;
uint16_t analog_input1_pin = 1;
uint16_t analog_input2_pin = 2;
uint16_t analog_input3_pin = 3;




uint16_t analog_input0 = 0;
uint16_t analog_input1 = 0;
uint16_t analog_input2 = 0;
uint16_t analog_input3 = 0;





uint16_t analog_input0_lp_filtered = 0;
uint16_t analog_input1_lp_filtered = 0;
uint16_t analog_input2_lp_filtered = 0;
uint16_t analog_input3_lp_filtered = 0;



uint16_t previous_analog_input0_lp_filtered = 0;
uint16_t previous_analog_input1_lp_filtered = 0;
uint16_t previous_analog_input2_lp_filtered = 0;
uint16_t previous_analog_input3_lp_filtered = 0;





// 50 Hz Butterworth low-pass
double a_lp_50Hz[] = {1.000000000000, -3.180638548875, 3.861194348994, -2.112155355111, 0.438265142262};
double b_lp_50Hz[] = {0.000416599204407, 0.001666396817626, 0.002499595226440, 0.001666396817626, 0.000416599204407};
IIRFilter lp_analog_input0(b_lp_50Hz, a_lp_50Hz);
IIRFilter lp_analog_input1(b_lp_50Hz, a_lp_50Hz);
IIRFilter lp_analog_input2(b_lp_50Hz, a_lp_50Hz);
IIRFilter lp_analog_input3(b_lp_50Hz, a_lp_50Hz);




//Thresholds for each sensor
uint16_t analog_input0_threshold = 75;
uint16_t analog_input1_threshold = 10;
uint16_t analog_input2_threshold = 5;
uint16_t analog_input3_threshold = 10;



/* IMU ***************************************************************************************************/

/* Set the delay between fresh samples */
static const unsigned long BNO055_PERIOD_MILLISECS = 100; // E.g. 4 milliseconds per sample for 250 Hz
//static const float BNO055_SAMPLING_FREQUENCY = 1.0e3f / PERIOD_MILLISECS;
#define BNO055_PERIOD_MICROSECS 100.0e3f //= 1000 * PERIOD_MILLISECS;
static uint32_t BNO055_last_read = 0;


Adafruit_BNO055 bno = Adafruit_BNO055(55); // Here set the ID. In this case it is 55. In this sketch the ID must be different from 0 as 0 is used to reset the EEPROM

bool reset_calibration = false;  // set to true if you want to redo the calibration rather than using the values stored in the EEPROM
bool display_BNO055_info = false; // set to true if you want to print on the serial port the infromation about the status and calibration of the IMU


/* Set the correction factors for the three Euler angles according to the wanted orientation */
float  correction_x = 0; // -177.19;
float  correction_y = 0; // 0.5;
float  correction_z = 0; // 1.25;




/* Displays some basic information on this sensor from the unified sensor API sensor_t type (see Adafruit_Sensor for more information) */
void displaySensorDetails(void)
{
    sensor_t sensor;
    bno.getSensor(&sensor);
    Serial.println("------------------------------------");
    Serial.print("Sensor:       "); Serial.println(sensor.name);
    Serial.print("Driver Ver:   "); Serial.println(sensor.version);
    Serial.print("Unique ID:    "); Serial.println(sensor.sensor_id);
    Serial.print("Max Value:    "); Serial.print(sensor.max_value); Serial.println(" xxx");
    Serial.print("Min Value:    "); Serial.print(sensor.min_value); Serial.println(" xxx");
    Serial.print("Resolution:   "); Serial.print(sensor.resolution); Serial.println(" xxx");
    Serial.println("------------------------------------");
    Serial.println("");
    delay(500);
}

/* Display some basic info about the sensor status */
void displaySensorStatus(void)
{
    /* Get the system status values (mostly for debugging purposes) */
    uint8_t system_status, self_test_results, system_error;
    system_status = self_test_results = system_error = 0;
    bno.getSystemStatus(&system_status, &self_test_results, &system_error);

    /* Display the results in the Serial Monitor */
    Serial.println("");
    Serial.print("System Status: 0x");
    Serial.println(system_status, HEX);
    Serial.print("Self Test:     0x");
    Serial.println(self_test_results, HEX);
    Serial.print("System Error:  0x");
    Serial.println(system_error, HEX);
    Serial.println("");
    delay(500);
}

/* Display sensor calibration status */
void displayCalStatus(void)
{
    /* Get the four calibration values (0..3) */
    /* Any sensor data reporting 0 should be ignored, */
    /* 3 means 'fully calibrated" */
    uint8_t system, gyro, accel, mag;
    system = gyro = accel = mag = 0;
    bno.getCalibration(&system, &gyro, &accel, &mag);

    /* The data should be ignored until the system calibration is > 0 */
    Serial.print("\t");
    if (!system)
    {
        Serial.print("! ");
    }

    /* Display the individual values */
    Serial.print("Sys:");
    Serial.print(system, DEC);
    Serial.print(" G:");
    Serial.print(gyro, DEC);
    Serial.print(" A:");
    Serial.print(accel, DEC);
    Serial.print(" M:");
    Serial.print(mag, DEC);
}

/* Display the raw calibration offset and radius data */
void displaySensorOffsets(const adafruit_bno055_offsets_t &calibData)
{
    Serial.print("Accelerometer: ");
    Serial.print(calibData.accel_offset_x); Serial.print(" ");
    Serial.print(calibData.accel_offset_y); Serial.print(" ");
    Serial.print(calibData.accel_offset_z); Serial.print(" ");

    Serial.print("\nGyro: ");
    Serial.print(calibData.gyro_offset_x); Serial.print(" ");
    Serial.print(calibData.gyro_offset_y); Serial.print(" ");
    Serial.print(calibData.gyro_offset_z); Serial.print(" ");

    Serial.print("\nMag: ");
    Serial.print(calibData.mag_offset_x); Serial.print(" ");
    Serial.print(calibData.mag_offset_y); Serial.print(" ");
    Serial.print(calibData.mag_offset_z); Serial.print(" ");

    Serial.print("\nAccel Radius: ");
    Serial.print(calibData.accel_radius);

    Serial.print("\nMag Radius: ");
    Serial.print(calibData.mag_radius);
}


/* Magnetometer calibration */
void performMagCal(void) {
  
  /* Get the four calibration values (0..3) */
  /* Any sensor data reporting 0 should be ignored, */
  /* 3 means 'fully calibrated" */
  uint8_t system, gyro, accel, mag;
  system = gyro = accel = mag = 0;
 
  while (mag != 3) {
    
    bno.getCalibration(&system, &gyro, &accel, &mag);
    if(display_BNO055_info){
      
      displayCalStatus();
      Serial.println("");
    }
  }
  
  if(display_BNO055_info){

    Serial.println("\nMagnetometer calibrated!");
  }
}  









/** Functions for handling received messages ***********************************************************************/

void receive_message() {
  
    static boolean reception_in_progress = false;
    static byte ndx = 0;
    char rcv_char;

    while (Serial.available() > 0 && new_message_received == false) {
        rcv_char = Serial.read();
        Serial.println(rcv_char);

        if (reception_in_progress == true) {
            if (rcv_char!= END_MARKER) {
                received_message[ndx] = rcv_char;
                ndx++;
                if (ndx >= MAX_LENGTH_MESSAGE) {
                    ndx = MAX_LENGTH_MESSAGE - 1;
                }
            }
            else {
                received_message[ndx] = '\0'; // terminate the string
                reception_in_progress = false;
                ndx = 0;
                new_message_received = true;
            }
        }
        else if (rcv_char == START_MARKER) {
            reception_in_progress = true;
        }
    }

    if (new_message_received) {
      handle_received_message(received_message);
      new_message_received = false;
    }
}







void handle_received_message(char *received_message) {

  //Serial.print("received_message: ");
  //Serial.println(received_message);

  
  char *all_tokens[2]; //NOTE: the message is composed by 2 tokens: command and value
  const char delimiters[5] = {START_MARKER, ',', ' ', END_MARKER,'\0'};
  int i = 0;

  all_tokens[i] = strtok(received_message, delimiters);
  
  while (i < 2 && all_tokens[i] != NULL) {
    all_tokens[++i] = strtok(NULL, delimiters);
  }

  char *command = all_tokens[0]; 
  char *value = all_tokens[1];


  if (strcmp(command,"motor1_pattern1") == 0 && strcmp(value,"1") == 0) {

    /*
    Serial.print("activating message 1: ");
    Serial.print(command);
    Serial.print(" ");
    Serial.print(value);
    Serial.println(" ");
    */
    
    analogWrite(digital_output3_pin, 200);
    
  }
  
  if (strcmp(command,"motor1_pattern1") == 0 && strcmp(value,"0") == 0) {

    /*
    Serial.print("activating message 2: ");
    Serial.print(command);
    Serial.print(" ");
    Serial.print(value);
    Serial.println(" ");
    */
    analogWrite(digital_output3_pin, 0);
    
  }


  
  if (strcmp(command,"LED1_pattern1") == 0) {

    /*
    Serial.print("activating message 1: ");
    Serial.print(command);
    Serial.print(" ");
    Serial.print(value);
    Serial.println(" ");
    */
    
    analogWrite(digital_output4_pin, atoi(value));
    
  }  
} 











/**************************************************************************************************************/

void setup() {
  Serial.begin(BAUD_RATE);
  while(!Serial);

  /* Setup of the digital sensors ******************************************************************************/
  
  pinMode(digital_input0_pin, INPUT);
  pinMode(digital_input1_pin, INPUT);
  pinMode(digital_input2_pin, INPUT);
  pinMode(digital_output0_pin, OUTPUT);
  pinMode(digital_output1_pin, OUTPUT);
  pinMode(digital_output2_pin, OUTPUT);
  pinMode(digital_output3_pin, OUTPUT);
  pinMode(digital_output4_pin, OUTPUT);


  digitalWrite(digital_output0_pin, HIGH);
  digitalWrite(digital_output1_pin, HIGH);
  digitalWrite(digital_output2_pin, HIGH);
  digitalWrite(digital_output3_pin, LOW);
  digitalWrite(digital_output4_pin, LOW);

  /* Setup of the analog sensors ******************************************************************************/
 
  // analogReadResolution(ANALOG_BIT_RESOLUTION); // Only for Teensy




  /* Setup of the IMU BNO055 sensor ******************************************************************************/
  
  /* Initialise the IMU BNO055 sensor */
  delay(1000);
  if (!bno.begin()){
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while (1);
  }

  int eeAddress = 0;
  long eeBnoID;
  long bnoID;
  bool foundCalib = false;

    
  if(reset_calibration){// Then reset the EEPROM so a new calibration can be made
    
    EEPROM.put(eeAddress, 0);
    eeAddress += sizeof(long);
    EEPROM.put(eeAddress, 0);
    eeAddress = 0;
    if(display_BNO055_info){
      Serial.println("EEPROM reset.");
      delay(10000);
    }
  }
  
  EEPROM.get(eeAddress, eeBnoID);
  
  adafruit_bno055_offsets_t calibrationData;
  sensor_t sensor;

  /*
  *  Look for the sensor's unique ID at the beginning oF EEPROM.
  *  This isn't foolproof, but it's better than nothing.
  */
  bno.getSensor(&sensor);
  bnoID = sensor.sensor_id;
    
  if (eeBnoID != bnoID) {
  
    if(display_BNO055_info){
      
      Serial.println("\nNo Calibration Data for this sensor exists in EEPROM");
      delay(2000);
    }
  }
  else{

    if(display_BNO055_info){  
       
      Serial.println("\nFound Calibration for this sensor in EEPROM.");
    }
    
    eeAddress += sizeof(long);
    EEPROM.get(eeAddress, calibrationData);

    if(display_BNO055_info){
      
      displaySensorOffsets(calibrationData);
      Serial.println("\n\nRestoring Calibration data to the BNO055...");
    }

    bno.setSensorOffsets(calibrationData);

    if(display_BNO055_info){
      
      Serial.println("\n\nCalibration data loaded into BNO055");
      delay(2000);
    }
    
    foundCalib = true;
  }

  if(display_BNO055_info){
    
    /* Display some basic information on this sensor */
    displaySensorDetails();

    /* Optional: Display current status */
    displaySensorStatus();

  }

  /* Crystal must be configured AFTER loading calibration data into BNO055. */
  bno.setExtCrystalUse(true);

  //sensors_event_t orientationData;
  //bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);

      
  
  if (foundCalib){
    
    performMagCal(); /* always recalibrate the magnetometers as it goes out of calibration very often */
  }
  else {
    
    if(display_BNO055_info){
      
      Serial.println("Please Calibrate Sensor: ");
      delay(2000); 
    }
        
    while (!bno.isFullyCalibrated()){

      if(display_BNO055_info){
        
            displayCalStatus();
            Serial.println("");
            delay(BNO055_PERIOD_MILLISECS); // Wait for the specified delay before requesting new data            
        }
    }

    adafruit_bno055_offsets_t newCalib;
    bno.getSensorOffsets(newCalib);
    
    if(display_BNO055_info){

      Serial.println("\nFully calibrated!");
      delay(3000);
      Serial.println("--------------------------------");
      Serial.println("Calibration Results: ");
    
      displaySensorOffsets(newCalib);

      Serial.println("\n\nStoring calibration data to EEPROM...");
    }


    eeAddress = 0;
    EEPROM.put(eeAddress, bnoID);
    eeAddress += sizeof(long);
    EEPROM.put(eeAddress, newCalib);


    if(display_BNO055_info){
      Serial.println("Data stored to EEPROM.");
      Serial.println("\n--------------------------------\n");
      delay(3000);
      }

  }
}




/****************************************************************************************************/

void loop() {


  receive_message();

  
  /* Loop for the analog and digital sensors ******************************************************************************/
  
  if (micros() - analog_last_read >= ANALOG_PERIOD_MICROSECS) {
    analog_last_read += ANALOG_PERIOD_MICROSECS;

    
    /* Loop for the analog sensors ******************************************************************************/

    analog_input0 = analogRead(analog_input0_pin);
    analog_input0_lp_filtered =  (uint16_t)lp_analog_input0.filter((double)analog_input0);
    analog_input1 = analogRead(analog_input1_pin);
    analog_input1_lp_filtered =  (uint16_t)lp_analog_input1.filter((double)analog_input1);
    analog_input2 = analogRead(analog_input2_pin);
    analog_input2_lp_filtered =  (uint16_t)lp_analog_input2.filter((double)analog_input2);                
    analog_input3 = analogRead(analog_input3_pin);
    analog_input3_lp_filtered =  (uint16_t)lp_analog_input3.filter((double)analog_input3);
    

    // Apply thresholds to the filtered signal
    analog_input0_lp_filtered = (analog_input0_lp_filtered < analog_input0_threshold) ? 0 : analog_input0_lp_filtered;
    analog_input1_lp_filtered = (analog_input1_lp_filtered < analog_input1_threshold) ? 0 : analog_input1_lp_filtered;
    analog_input2_lp_filtered = (analog_input2_lp_filtered < analog_input2_threshold) ? 0 : analog_input2_lp_filtered;
    analog_input3_lp_filtered = (analog_input3_lp_filtered < analog_input3_threshold) ? 0 : analog_input3_lp_filtered;



    //Plot on the Serial Plotter the unfiltered sensors values 
    /*
    Serial.print(analog_input0);
    Serial.print(" ");
    Serial.print(analog_input1);
    Serial.print(" ");
    Serial.print(analog_input2);
    Serial.print(" ");
    Serial.println(analog_input3);
    */


    //Plot on the Serial Plotter the filtered sensors values 
    /*
    Serial.print(analog_input0_lp_filtered);
    Serial.print(" ");
    Serial.print(analog_input1_lp_filtered);
    Serial.print(" ");
    Serial.print(analog_input2_lp_filtered);
    Serial.print(" ");
    Serial.println(analog_input3_lp_filtered);
    */



    // Send the sensor value to the serial port only if it has changed
    
    if(analog_input0_lp_filtered != previous_analog_input0_lp_filtered){
      Serial.print("a0, ");
      Serial.println(analog_input0_lp_filtered);
      //Serial.print(analog_input0);
      //Serial.print(" ");
      //Serial.println(analog_input0_lp_filtered);
      previous_analog_input0_lp_filtered = analog_input0_lp_filtered;
    }
    if(analog_input1_lp_filtered != previous_analog_input1_lp_filtered){
      Serial.print("a1, ");
      Serial.println(analog_input1_lp_filtered);
      //Serial.print(analog_input1);
      //Serial.print(" ");
      //Serial.println(analog_input1_lp_filtered);
      previous_analog_input1_lp_filtered = analog_input1_lp_filtered;
    }
    if(analog_input2_lp_filtered != previous_analog_input2_lp_filtered){
      Serial.print("a2, ");
      Serial.println(analog_input2_lp_filtered);
      previous_analog_input2_lp_filtered = analog_input2_lp_filtered;
    }
    if(analog_input3_lp_filtered != previous_analog_input3_lp_filtered){
      Serial.print("a3, ");
      Serial.println(analog_input3_lp_filtered);
      previous_analog_input3_lp_filtered = analog_input3_lp_filtered;
    }
    

        
  } // End of the section processing the analog sensors with the set sample rate for them. Such sample rate is different from that of the IMU



  /* Loop for the IMU BNO055 sensor ******************************************************************************/
  
  if (micros() - BNO055_last_read >= BNO055_PERIOD_MICROSECS) {
    BNO055_last_read += BNO055_PERIOD_MICROSECS;
  
    sensors_event_t orientationData, angVelData, linearAccelData;
    bno.getEvent(&orientationData, Adafruit_BNO055::VECTOR_EULER);
    bno.getEvent(&angVelData, Adafruit_BNO055::VECTOR_GYROSCOPE);
    bno.getEvent(&linearAccelData, Adafruit_BNO055::VECTOR_LINEARACCEL);
    
    
    /*
     Note:
     x = Yaw, y = Roll, z = pitch 
     
     The Yaw values are between 0° to +360°
     The Roll values are between -90° and +90°
     The Pitch values are between -180° and +180°
    */ 
    
    /*
    Serial.print(": Euler x= ");
    Serial.print(orientationData.orientation.x + correction_x); // I add a correction value to get the 0 for the orientation I need 
    Serial.print(" | Euler y= ");
    Serial.print(orientationData.orientation.y + correction_y);
    Serial.print(" | Euler z= ");
    Serial.print(orientationData.orientation.z + correction_z);
    */

    Serial.print("x, ");
    Serial.println(orientationData.orientation.x + correction_x); // I add a correction value to get the 0 for the orientation I need 
    Serial.print("y, ");
    Serial.println(orientationData.orientation.y + correction_y); // I add a correction value to get the 0 for the orientation I need 
    Serial.print("z, ");
    Serial.println(orientationData.orientation.z + correction_z); // I add a correction value to get the 0 for the orientation I need 
     

    /*
    Serial.print(": angVel x= ");
    Serial.print(angVelData.gyro.x);
    Serial.print(" | angVel y= ");
    Serial.print(angVelData.gyro.y);
    Serial.print(" | angVel z= ");
    Serial.print(angVelData.gyro.z);

    Serial.print(": linearAccel x= ");
    Serial.print(linearAccelData.acceleration.x);
    Serial.print(" | linearAccel y= ");
    Serial.print(linearAccelData.acceleration.y);
    Serial.print(" | linearAccel z= ");
    Serial.print(linearAccelData.acceleration.z);
    */

    
    //Serial.println("");
    
    /* Optional: Display calibration status */
    //displayCalStatus();
    /* Optional: Display sensor status (debug only) */
    //displaySensorStatus();
    /* New line for the next sample */
    //Serial.println("");

    }//End loop IMU
    
    
    
    
    
    /* Loop for the digital sensors ******************************************************************************/

    
    /* Handle button_0-LED_0 ********************************/
    digital_input0 = digitalRead(digital_input0_pin);
    
    if (digital_input0 != digital_input0_last_button_state) {
      // reset the debouncing timer
      digital_input0_last_debounce_time = millis();
    }
    
    if ((millis() - digital_input0_last_debounce_time) > debounce_delay) {
      
      // if the button state has changed:
      if (digital_input0 != digital_input0_button_state) {
        digital_input0_button_state = digital_input0;
    
        Serial.print("d0, ");
        Serial.println(digital_input0_button_state);
    
        // only toggle the LED if the new button state is HIGH
        if (digital_input0_button_state == HIGH) {
          digital_output0_LED_state = !digital_output0_LED_state;
        }
      }
    }
    
    digitalWrite(digital_output0_pin, digital_output0_LED_state);
    digital_input0_last_button_state = digital_input0;
    
    
            
    
    
    
    /* Handle button_1-LED_1 ********************************/
    digital_input1 = digitalRead(digital_input1_pin);
    
    if (digital_input1 != digital_input1_last_button_state) {
      // reset the debouncing timer
      digital_input1_last_debounce_time = millis();
    }
    
    if ((millis() - digital_input1_last_debounce_time) > debounce_delay) {
    
      // if the button state has changed:
      if (digital_input1 != digital_input1_button_state) {
        digital_input1_button_state = digital_input1;
        Serial.print("d1, ");
        Serial.println(digital_input1_button_state);
    
        // only toggle the LED if the new button state is HIGH
        if (digital_input1_button_state == HIGH) {
          digital_output1_LED_state = !digital_output1_LED_state;
        }
      }
    }
    
    digitalWrite(digital_output1_pin, digital_output1_LED_state);
    digital_input1_last_button_state = digital_input1;
    
    
    /* Handle button_2-LED_2 ********************************/
    digital_input2 = digitalRead(digital_input2_pin);
    
    if (digital_input2 != digital_input2_last_button_state) {
      // reset the debouncing timer
      digital_input2_last_debounce_time = millis();
    }
    
    if ((millis() - digital_input2_last_debounce_time) > debounce_delay) {
    
      // if the button state has changed:
      if (digital_input2 != digital_input2_button_state) {
        digital_input2_button_state = digital_input2;
        Serial.print("d2, ");
        Serial.println(digital_input2_button_state);
    
        // only toggle the LED if the new button state is HIGH
        if (digital_input2_button_state == HIGH) {
          digital_output2_LED_state = !digital_output2_LED_state;
        }
      }
    }
    
    digitalWrite(digital_output2_pin, digital_output2_LED_state);
    digital_input2_last_button_state = digital_input2;  
  
}
