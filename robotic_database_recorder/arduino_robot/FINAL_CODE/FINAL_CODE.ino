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
#define FULLSTEP 4
#define STEP_PER_REVOLUTION 2048 // this value is from datasheet


#define CONTROLLER_TOP_PIN1 2
#define CONTROLLER_TOP_PIN2 3
#define CONTROLLER_TOP_PIN3 4
#define CONTROLLER_TOP_PIN4 5

#define CONTROLLER_RIGHT_PIN1 6
#define CONTROLLER_RIGHT_PIN2 7
#define CONTROLLER_RIGHT_PIN3 8
#define CONTROLLER_RIGHT_PIN4 9


#define BUTTON_A_PIN 10
#define BUTTON_B_PIN 11

#define LED_LA_PIN 12
#define LED_LB_PIN 13


#define STEPPER_MAXSPEED 50.0
#define STEPPER_ACCELERATION 10.0
#define STEPPER_SPEED 50.0

// Declare Stepper Controllers
AccelStepper stepperTop(FULLSTEP, CONTROLLER_TOP_PIN1, CONTROLLER_TOP_PIN2, CONTROLLER_TOP_PIN3, CONTROLLER_TOP_PIN4);
AccelStepper stepperRight(FULLSTEP, CONTROLLER_RIGHT_PIN1, CONTROLLER_RIGHT_PIN2, CONTROLLER_RIGHT_PIN3, CONTROLLER_RIGHT_PIN4);


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

void setup() {
  Serial.begin(9600);
  stepperTop.setMaxSpeed(STEPPER_MAXSPEED);   // set the maximum speed
  stepperTop.setAcceleration(STEPPER_ACCELERATION); // set acceleration
  stepperTop.setSpeed(STEPPER_SPEED);         // set initial speed
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
}

void setZeroPosition () {
  stepperTop.setCurrentPosition(0); // set position
  stepperRight.setCurrentPosition(0); // set position
  // Serial.print("Stepper now at pos ");
  // Serial.print(stepper.currentPosition());
  // Serial.println("/1024");
}

/**
 * Converts knob position [0,10] to steps for the stepper controller
*/
float pos2step (float pos, float full_rotation_degrees) {
  if (pos<=10) {
    return pos/10.0*STEP_PER_REVOLUTION/360.0*full_rotation_degrees;
  }
  return -1.0f;
}

// TO move the steppers
// stepperTop.moveTo(newpos);
// stepperRight.moveTo(newpos);

void loop() {
  stepperTop.run();
  stepperRight.run();
}
