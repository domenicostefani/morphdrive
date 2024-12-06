/**
* Edited from the Multisensory Interactive Systems" course at the University of Trento, Italy
* Author: Luca Turchet
* Date: 30/05/2019
* 
**/

// #include <EEPROM.h> // TODO: maybe use to store stepper calibration data (Beginning of potentiometer range, end of potentiometer range)
#include <string.h>

#define BAUD_RATE 115200

/* Variables for incoming messages *************************************************************/

const byte MAX_LENGTH_MESSAGE = 64;
char received_message[MAX_LENGTH_MESSAGE];

char START_MARKER = '[';
char END_MARKER = ']';
    
boolean new_message_received = false;

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


    if (strcmp(command,"all_pots_to_min") == 0 && strcmp(value,"1") == 0) {
        // TODO: implement the function to set all the pots to the minimum value 
    }

    if (strcmp(command,"all_pots_to_max") == 0 && strcmp(value,"1") == 0) {
        // TODO: implement the function to set all the pots to the maximum value
    }

    if (strcmp(command,"potgain_pos") == 0) {
        // TODO: implement control of relative pot with atoi(value), maybe lock if calibration was not done
    }  

    if (strcmp(command,"pottone_pos") == 0) {
        // TODO: implement control of relative pot with atoi(value), maybe lock if calibration was not done
    }
} 

/**************************************************************************************************************/

void setup() {
    Serial.begin(BAUD_RATE);
    while(!Serial);
}

/****************************************************************************************************/

void loop() {
    receive_message();
    
}
