/*
 * Created by ArduinoGetStarted.com
 *
 * This example code is in the public domain
 *
 * Tutorial page: https://arduinogetstarted.com/tutorials/arduino-controls-28byj-48-stepper-motor-using-uln2003-driver
 */

// Include the AccelStepper Library
#include <AccelStepper.h>

#define POT_ROTATION_DEGREES 300

// define step constant
#define FULLSTEP 4
#define STEP_PER_REVOLUTION 2048 // this value is from datasheet

// Pins entered in sequence IN1-IN3-IN2-IN4 for proper step sequence
AccelStepper stepper(FULLSTEP, 5, 4, 3, 2);

void setup() {
  Serial.begin(9600);
  stepper.setMaxSpeed(50.0);   // set the maximum speed
  stepper.setAcceleration(10.0); // set acceleration
  stepper.setSpeed(50.0);         // set initial speed
  stepper.setCurrentPosition(0); // set position
  // stepper.moveTo(STEP_PER_REVOLUTION); // set target position: 64 steps <=> one revolution

  Serial.print("Stepper now at pos ");
  Serial.print(stepper.currentPosition());
  Serial.println("/1024");
}

float serialNewPos = 0;
int lastpos = 0;

void loop() {
  // change direction once the motor reaches target position
  // if (stepper.distanceToGo() == 0)
  //   stepper.moveTo(-stepper.currentPosition());
  
  while (Serial.available() > 0) {
    float tmp = Serial.parseInt();

    if ((tmp<=10)&&(Serial.read() == '\n')) {
      float newpos = tmp/10.0*STEP_PER_REVOLUTION/360.0*300.0;
      Serial.print("I will move to ");
      Serial.println(newpos);
      stepper.moveTo(newpos);
    }
  }

  stepper.run(); // MUST be called in loop() function

  // Serial.print(F("Current Position: "));
  if (stepper.currentPosition() != lastpos) {
    lastpos = stepper.currentPosition();
    Serial.println(lastpos);
  }
}
