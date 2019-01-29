
#include <vexMotor.h> 
vexMotor myVexMotor1;  


void setup()
{
  myVexMotor1.attach(9); // setup / attach the vexMotor onto pin 9
  Serial.begin(9600);    // starts the Serial communication on Arduino
}

void loop()
{
  int inputValue;
  

  // Read in value from Serial Monitor
   while (Serial.available() == 0)
  {
  }

  inputValue = Serial.parseInt();
  Serial.write(inputValue);
  
  if(inputValue == 1)
  {
  myVexMotor1.write(35);
    delay(150);
    myVexMotor1.write(0);
    }

    
   if(inputValue == 2) 
    {
    myVexMotor1.write(-35);
    delay(150);
    myVexMotor1.write(0);
   }


}
