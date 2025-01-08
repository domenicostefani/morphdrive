Robot Recorder
---

Robotic guitar pedal dataset recorder.  

### Operation
The robot has two buttons (__A__, __B__), two LEDs (__LA__, __LB__), and two motors (__TOP__, __RIGHT__).

1. Connect gears to the GAIN and TONE knobs of the guitar pedal, and set both to zero (fully counter-clockwise).
2. Set the pedal on the velcro pad, attach pulleys to knobs and stepper motors. GAIN goes to the TOP motor, TONE goes to the RIGHT motor.
3. Connect the robot to the computer, and run the PureData patch.

#### Calibration
1. Press Button __A__ for 3 seconds to enter _RESET MODE_. __LA__ will turn on and this will set the zero position for the motors/knobs.
2. Press B and the __TOP__ motor will start moving. Keep it pressed and release when the motor reaches the end of the range.
3. Press Buttons __A__ and __B__ together to set the end position for the __TOP__ motor.
4. Press Button __B__ to until motor __RIGHT__ reaches the end of the range. Press Buttons __A__ and __B__ together for 3 seconds to set the end position for the __RIGHT__ motor. 
5. LA will turn off, both motors will move to the zero position.

#### Recording
1. Start recording from the PureData patch. The robot will start moving the motors and record the data from the guitar pedal. At the end of the recording, the robot will move the motors back to the zero position. If not, press Button __B__.
2. Switch out the pedal. __MAKE SURE__ the knobs are set to zero before starting the recording.








### Status
- PureData Recorder: Made and tested with PlugData v0.9.1  
- Robot:
    - Arduino code: todo
    - Hardware: todo

### Recorder
<img src="https://github.com/user-attachments/assets/f21e7fb9-e6d1-4a20-9755-fd3c01ef1319" width="70%" >  
