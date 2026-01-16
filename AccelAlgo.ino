#include <AccelStepper.h>
#define DRIVE AccelStepper::DRIVER
#define FULL4 AccelStepper::FULL4WIRE
String inputString = "";
bool stringComplete = false;

AccelStepper X(DRIVE,2,3);
AccelStepper Z(DRIVE,30,31);
AccelStepper R(FULL4,50,52,51,53);
AccelStepper Y1(FULL4,4,5,6,7);
AccelStepper Y2(FULL4,9,10,11,12);

long M, x, y , z = 0;
long r = 0;
long ZSTEP = 0;
long YSTEP = 0;
long XSTEP = 0;
long RSTEP = 0;
int mode = 1;
void setup() {
  Serial.begin(115200);


  Z.setMaxSpeed(300);
  Z.setAcceleration(100);
  Z.moveTo(ZSTEP);


  Y1.setMaxSpeed(300);
  Y1.setAcceleration(100);
  Y1.moveTo(YSTEP);

  Y2.setMaxSpeed(300);
  Y2.setAcceleration(100);
  Y2.moveTo(YSTEP);

  X.setMaxSpeed(900);
  X.setAcceleration(300);
  X.moveTo(XSTEP);

  R.setMaxSpeed(300);
  R.setAcceleration(100);
  R.moveTo(RSTEP);
}

void loop() {
  serialEvent();
  if (stringComplete){

  

  sscanf(inputString.c_str(), "M%ld X%ld Y%ld Z%ld R%ld",&M, &x, &y, &z, &r);

  

  XSTEP =(long) ((x*800)/1.25);
  YSTEP =(long) ((y*200)/1.25); 
  ZSTEP =(long) ((z*200)/1.25);
  RSTEP = (long) (r*2052)/360;
 

  inputString = "";
  stringComplete = false;

  X.moveTo(XSTEP);
  Z.moveTo(ZSTEP);
  Y1.moveTo(YSTEP);
  Y2.moveTo(YSTEP);
  R.moveTo(RSTEP);
  } 
switch (M){
    case 0 :
      motorControl(); 
      break;

    case 1:
      X.disableOutputs();Y1.disableOutputs();Y2.disableOutputs();Z.disableOutputs();R.disableOutputs();
      break;
    case 2:
      X.enableOutputs();Y1.enableOutputs();Y2.enableOutputs();Z.enableOutputs();R.enableOutputs();
      break;


  
}
 
  
  }



void motorControl(){

 X.run();
 
 if (!X.isRunning()){X.disableOutputs();}
 else{X.enableOutputs();};

 Y1.run();

 Y2.run();
 if (!Y1.isRunning()){ Y1.disableOutputs();Y2.disableOutputs();}
 else{Y1.enableOutputs();Y2.enableOutputs();};

 Z.run();

 if (!Z.isRunning()){Z.disableOutputs();}
 else{Z.enableOutputs();};
 
 R.run();

 if (!R.isRunning()){R.disableOutputs();}
 else{R.enableOutputs();};


 if ((!X.isRunning()) && (!Y1.isRunning()) && (!Y2.isRunning()) && (!Z.isRunning()) && (!R.isRunning())){Serial.println("Done");M=3;};
  

}
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;}
}
}