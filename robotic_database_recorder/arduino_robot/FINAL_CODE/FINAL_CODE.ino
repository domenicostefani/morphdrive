/*
 * Stepper code based on: https://arduinogetstarted.com/tutorials/arduino-controls-28byj-48-stepper-motor-using-uln2003-driver
 */

// Include the AccelStepper Library
#include <AccelStepper.h>

// Turns off all OSC code and prints calibration and sensor data to serial
#define DEBUG_MODE true 
// // De
// #define POT_ROTATION_DEGREES 300

// define step constant
#define STEP_PER_REVOLUTION_TOP 2038
#define STEP_PER_REVOLUTION_RIGHT 509

#define CONTROLLER_TOP_PIN1 6
#define CONTROLLER_TOP_PIN2 7
#define CONTROLLER_TOP_PIN3 8
#define CONTROLLER_TOP_PIN4 9

#define CONTROLLER_RIGHT_PIN1 2
#define CONTROLLER_RIGHT_PIN2 3
#define CONTROLLER_RIGHT_PIN3 4
#define CONTROLLER_RIGHT_PIN4 5

#define BUTTON_A_PIN 11
#define BUTTON_B_PIN 10

#define LED_LA_PIN 13
#define LED_LB_PIN 12


#define STEPPER_MAXSPEED 20.0
#define STEPPER_ACCELERATION 10.0
#define STEPPER_SPEED 20.0

// Declare Stepper Controllers
AccelStepper stepperTop(AccelStepper::FULL4WIRE, CONTROLLER_TOP_PIN1, CONTROLLER_TOP_PIN3, CONTROLLER_TOP_PIN2, CONTROLLER_TOP_PIN4);
AccelStepper stepperRight(AccelStepper::FULL4WIRE, CONTROLLER_RIGHT_PIN1, CONTROLLER_RIGHT_PIN3, CONTROLLER_RIGHT_PIN2, CONTROLLER_RIGHT_PIN4);

#include <string.h>

#define BAUD_RATE 115200

/* Variables for incoming messages *************************************************************/

const byte MAX_LENGTH_MESSAGE = 64;
char received_message[MAX_LENGTH_MESSAGE];

char START_MARKER = '[';
char END_MARKER = ']';
    
boolean new_message_received = false;

/** Functions for handling received messages ***********************************************************************/


/**
 * Converts knob position [0,10] to steps for the stepper controller
*/
float pos2step (float pos, int steps_per_revolution, float full_rotation_degrees=300) {
  if ((pos<=1.0)&&(pos>=0)) {
    return pos*steps_per_revolution/360.0*full_rotation_degrees;
  }
  return -1.0f;
}

void receive_message() {
  
    static boolean reception_in_progress = false;
    static byte ndx = 0;
    char rcv_char;

    while (Serial.available() > 0 && new_message_received == false) {
        rcv_char = Serial.read();
        // Serial.println(rcv_char);

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

void blink(int pin) {
  digitalWrite(pin, HIGH);
    delay(100);
    digitalWrite(pin, LOW);
    delay(100);
}

void handle_received_message(char *received_message) {
    Serial.print("received_message: ");
    Serial.println(received_message);

    
    

    char *all_tokens[2]; //NOTE: the message is composed by 2 tokens: command and value
    const char delimiters[5] = {START_MARKER, ',', ' ', END_MARKER,'\0'};
    int i = 0;

    all_tokens[i] = strtok(received_message, delimiters);

    while (i < 2 && all_tokens[i] != NULL) {
        all_tokens[++i] = strtok(NULL, delimiters);
    }

    char *command = all_tokens[0]; 
    char *value = all_tokens[1];


    // Serial.print("command:");
    // Serial.println(command);

    // Serial.print("value: ");
    // Serial.println(value);

    if (strcmp(command,"all_pots_to_min") == 0 && strcmp(value,"1") == 0) {
        // TODO: implement the function to set all the pots to the minimum value
        stepperTop.moveTo(0);
        stepperRight.moveTo(0);
        blink(LED_LB_PIN);
        blink(LED_LB_PIN);
    }

    if (strcmp(command,"all_pots_to_max") == 0 && strcmp(value,"1") == 0) {
        // TODO: implement the function to set all the pots to the maximum value
        stepperTop.moveTo(pos2step(1.0,STEP_PER_REVOLUTION_TOP));
        stepperRight.moveTo(pos2step(1.0,STEP_PER_REVOLUTION_RIGHT));
        blink(LED_LB_PIN);
        blink(LED_LB_PIN);
        blink(LED_LB_PIN);
    }

    if (strcmp(command,"top_pos") == 0) {
        float result = atof(value);
        if ((result >= 0.0) || (result <= 1.0))
        {
          stepperTop.moveTo(pos2step(result,STEP_PER_REVOLUTION_TOP));
          blink(LED_LB_PIN);
        }
    }  

    if (strcmp(command,"right_pos") == 0) {
        float result = atof(value);
        if ((result >= 0.0) || (result <= 1.0))
        {
          stepperRight.moveTo(pos2step(result,STEP_PER_REVOLUTION_RIGHT));
          blink(LED_LB_PIN);
        }
    }
}

class DebouncedButton {
  const unsigned long debounceDelay = 50; // Debounce time in milliseconds
  int pin;
  int buttonState = LOW;   // Current state of the button
  int lastButtonState = LOW; // Previous state of the button
  unsigned long lastDebounceTime = 0; // Last time the button state changed
public:
  DebouncedButton(int pin) {
    this->pin = pin;
  }

  void setup() {
    pinMode(pin, INPUT_PULLUP);
  }

  void loop(){
    int reading = digitalRead(pin); // Read the button state

    // Check if the button state has changed
    if (reading != lastButtonState) {
      lastDebounceTime = millis(); // Reset the debounce timer
      return;
    }
  

    // If the state has been stable for debounceDelay milliseconds, update the button state
    if ((millis() - lastDebounceTime) > debounceDelay) {
      if (reading != buttonState) {
        buttonState = reading;

        // change the inner state
        if (buttonState == LOW) {
          buttonState = HIGH;
        } else {
          buttonState = LOW;
        }
      }
    }
  }

  bool getPressed(){
    return buttonState;
  }
};


DebouncedButton buttonA{BUTTON_A_PIN};
DebouncedButton buttonB{BUTTON_B_PIN};

enum SysState {
  NO_CALIBRATION = 0,
  CALIBRATION_MODE,
  OPERATIVE
} systemState;



void setZeroPosition () {
  stepperTop.setCurrentPosition(0); // set position
  stepperRight.setCurrentPosition(0); // set position
  // Serial.print("Stepper now at pos ");
  // Serial.print(stepper.currentPosition());
  // Serial.println("/1024");
}

void setup() {
  Serial.begin(BAUD_RATE);
  while(!Serial);
  // 4 times more because slower gear ratio on top
  stepperTop.setMaxSpeed(STEPPER_MAXSPEED*4.0);   // set the maximum speed
  stepperTop.setAcceleration(STEPPER_ACCELERATION*4.0); // set acceleration
  stepperTop.setSpeed(STEPPER_SPEED*4.0);         // set initial speed

  stepperRight.setMaxSpeed(STEPPER_MAXSPEED);   // set the maximum speed
  stepperRight.setAcceleration(STEPPER_ACCELERATION); // set acceleration
  stepperRight.setSpeed(STEPPER_SPEED);         // set initial speed

  pinMode(CONTROLLER_TOP_PIN1, OUTPUT);
  pinMode(CONTROLLER_TOP_PIN2, OUTPUT);
  pinMode(CONTROLLER_TOP_PIN3, OUTPUT);
  pinMode(CONTROLLER_TOP_PIN4, OUTPUT);
  pinMode(CONTROLLER_RIGHT_PIN1, OUTPUT);
  pinMode(CONTROLLER_RIGHT_PIN2, OUTPUT);
  pinMode(CONTROLLER_RIGHT_PIN3, OUTPUT);
  pinMode(CONTROLLER_RIGHT_PIN4, OUTPUT);

  buttonA.setup();
  buttonB.setup();

  pinMode(LED_LA_PIN, OUTPUT);
  pinMode(LED_LB_PIN, OUTPUT);

  setZeroPosition();
  
  // stepperRight.moveTo(pos2step(1,STEP_PER_REVOLUTION_RIGHT));
  // stepperTop.moveTo(pos2step(1,STEP_PER_REVOLUTION_TOP));
}


// TO move the steppers
// stepperTop.moveTo(newpos);
// stepperRight.moveTo(newpos);

void loop() {
  receive_message();
  stepperTop.run();
  stepperRight.run();
}
