import oscP5.*;
import netP5.*;
import controlP5.*;

PImage backgroundImg;
PSNColors psnColors;
PedalPoints pedalPoints;
ArrayList<PVector> centers;
float SCALE; // Gui scale, from [0,1] coordinates to [width,height] pixel coordinates to display
PVector mousePos;

boolean drawPointer = true, rendering = false;
OscP5 oscP5;
NetAddress pythonReceiver;
int OSC_SEND_EVERY_X_FRAMES = 6; // with framerate at 60fps, osc framerate is 10fps


ControlP5 controlP5;
Knob[] latKnobs;


void setup() {
  size(720, 640);
  SCALE = min(width, height); // Set scale to minimum window dimension
  frameRate(60);

  // Initialize OSC
  oscP5 = new OscP5(this, 12000); // Listen for incoming OSC messages on port 12000 (if needed) //<>//
  pythonReceiver = new NetAddress("127.0.0.1", 12345); // Python receiver on localhost:12345

  pedalPoints = new PedalPoints();
  psnColors = new PSNColors();

  pedalPoints.test_randomInit(50); // generate 50 test points spanning 4 classes


  /**
   * Render Background once
   */
  renderBackground();

  drawPointer = true;//TODO: remove this line

  mousePos = new PVector(); // Save mouse position
  
  // Ask for dataframe with pedal points
  OscMessage msg = new OscMessage("/gimmeDataframe");
  oscP5.send(msg, pythonReceiver);
  
  
  controlP5 = new ControlP5(this);
  int margin = int((width-height)*0.12);
  latKnobs = new Knob[8];
  for (int pix=0; pix <8; pix++){
    latKnobs[pix] = controlP5.addKnob("latent"+str(pix),
                              -1,//min
                              1, //max
                              0, //default
                              height+margin, //x
                              margin+pix*((height-2*margin)/8), //y
                              width-height-2*margin); //width
    float alpha = 1;
    latKnobs[pix].setColorBackground(psnColors.getOpaque(pix, alpha));  // Set foreground color to white
    float randomval = random(-1.0,1.0);
    latKnobs[pix].setValue(randomval);
  }
}

boolean DO_RENDER_BACKGROUND = false;

void renderBackground() {
  if (!rendering) {
    rendering = true;

    centers = pedalPoints.computeCenters(); //<>//
    drawBackground(SCALE); // Draw Background with (1) measurement points (2) faded color area blobs (3) area barycenters
 //<>// //<>// //<>// //<>//
    save("back.png"); // Save as an image to save computations later in draw()
    backgroundImg = loadImage("back.png");  // Load into backgroundImg //<>//

    rendering = false;
  }
}

void drawTooltip(float x, float y, float val1, float val2) {
    // draw a tooltip that should look like a minimal baloon
    // First there should be a triangle pointing up to x,t, with equal sides at 20px
    // Then there should be a rectangle with rounded corners of 5px, attached to the bottom 
    // of the triangle, with a width of 100px and a height of 50px
    y = y+15; // move tooltip down a bit
    fill(255, 255, 255, 90);
    noStroke();
    int val1_10 = round(val1*10.0);
    int val2_10 = round(val2*10.0);
    // stroke(255, 255, 255, 50);
    // Draw triangle
    beginShape();
    vertex(x, y);
    vertex(x-10, y+21);
    vertex(x+10, y+21);
    endShape(CLOSE);

    // Draw rectangle
    float rectWidth = 100, rectHeight = 50, bottomTextArea = 14;
    rect(x-rectWidth/2, y+20, rectWidth, rectHeight+bottomTextArea, 5);
    PVector leftCenter = new PVector(rectWidth/4.0+x-rectWidth/2, rectHeight/2.0+y+20);
    PVector rightCenter = new PVector(3.0*rectWidth/4.0+x-rectWidth/2, rectHeight/2.0+y+20);

    // We now draw a knob in each center
    noFill();
    stroke(0);
    float margin = 5;
    float potRadius = min(rectHeight/2.0-margin, rectWidth/4.0-margin);
    ellipse(leftCenter.x, leftCenter.y, potRadius*2, potRadius*2);
    ellipse(rightCenter.x, rightCenter.y, potRadius*2, potRadius*2);

    // We now draw the values of the knobs with a line. value 0 should be at -150 degrees from the top, value 1 at 150 degrees from the top
    
    float startangle = -240;
    float endangle = 60; 
    float angle1 = radians(map(val1, 0, 1,startangle, endangle));
    float angle2 = radians(map(val2, 0, 1, startangle, endangle));
    PVector leftKnob = PVector.add(leftCenter, PVector.fromAngle(angle1).mult(potRadius));
    PVector rightKnob = PVector.add(rightCenter, PVector.fromAngle(angle2).mult(potRadius));
    line(leftCenter.x, leftCenter.y, leftKnob.x, leftKnob.y);
    line(rightCenter.x, rightCenter.y, rightKnob.x, rightKnob.y);

    // Now draw Gain and Tone labels under the knobs
    fill(0);
    textAlign(CENTER, CENTER);
    textSize(12);
    // text("Gain: "+int(val1*10), leftCenter.x, leftCenter.y+potRadius+10);
    // text("Tone: "+int(val2*10), rightCenter.x, rightCenter.y+potRadius+10);
    text("Gain: "+(val1_10), leftCenter.x, leftCenter.y+potRadius+10);
    text("Tone: "+(val2_10), rightCenter.x, rightCenter.y+potRadius+10);


    // text("G: "+val1, x-40, y+40);
    // text("T: "+val2, x-40, y+60);


}

long lastCheck = 0;
boolean lockTooltip = false;

void draw() {
  // Draw rendered background
  if (DO_RENDER_BACKGROUND) {
    renderBackground();
    DO_RENDER_BACKGROUND = false;
  } else {
    image(backgroundImg, 0, 0);
  }

  fill(0);
  stroke(255);
  
  //text("mousex: "+str(mousePos.x), 100, 100);

  boolean isInsidePlayground = (mouseX < height) && (mouseY < height);
  // Allow locking cursor position with mouse press
  if (!cursorLock && isInsidePlayground) {
    mousePos.x = mouseX;
    mousePos.y = mouseY;
    noCursor();
  } else if (cursorLock){
    cursor(CROSS);
  } else {
    cursor(HAND);
  }

  if (centers.size() == 0)
    drawPointer = false;

  // Draw pot-shaped cursor
  PVector scaledMousePos = mousePos.copy().div(SCALE);
  if (drawPointer) {
    float[] closestAreaIdxDist = getClosestCenter(scaledMousePos);
    int closestAreaIdx = int(closestAreaIdxDist[0]);
    float closestAreaDist = closestAreaIdxDist[1];
    float alpha = map(closestAreaDist, 0, 0.5, 255, 0);
    color potColor = psnColors.getOpaque(closestAreaIdx, alpha);
    int potRadius = 15;
    // Actual drawing circle
    fill(potColor);
    stroke(0);
    ellipse(mousePos.x, mousePos.y, potRadius*2, potRadius*2);
    // Draw pot intex pointing to closest area
    if ((closestAreaDist >= 0) && (closestAreaIdx >= 0)) {
      PVector directionVector = PVector.sub(centers.get(closestAreaIdx), scaledMousePos).normalize();
      PVector intersectionPoint = PVector.add(mousePos, PVector.mult(directionVector, potRadius));
    
      // Actual pot-index drawing
      line(mousePos.x, mousePos.y, intersectionPoint.x, intersectionPoint.y);
    }
  }

  // Send OSC message
  if (frameCount % OSC_SEND_EVERY_X_FRAMES == 0) {
    sendOscMessage(scaledMousePos);
  }

  // Draw tooltip if hovering any point
  if ((millis() > lastCheck + 500)||lockTooltip) {
    // lastCheck = millis();
    PedalPoint hoveringPoint = pedalPoints.getHovering(scaledMousePos);
    if (hoveringPoint != null) {
      drawTooltip(hoveringPoint.getX()*SCALE, hoveringPoint.getY()*SCALE, hoveringPoint.getGain(), hoveringPoint.getTone());
      lockTooltip = true;
    } else {
      lockTooltip = false;
    }
  }
}

void mousePressed() {
  // Lock cursor position when pressing
  cursorLock = !cursorLock;
}


void sendOscMessage(PVector mousepos01) {
  OscMessage msg = new OscMessage("/mouse/positionScaled");
  msg.add(mousepos01.x);
  msg.add(mousepos01.y);
  oscP5.send(msg, pythonReceiver);
}

void oscEvent(OscMessage theOscMessage) {
  /* check if theOscMessage has the address pattern we are looking for. */

  if (theOscMessage.checkAddrPattern("/clearPoints")==true) {
    if (theOscMessage.checkTypetag("")) {
      println("$ I was asked to clear points");
      pedalPoints.clear();
      return;
    }
  }
  if (theOscMessage.checkAddrPattern("/renderBackground")==true) {
    if (theOscMessage.checkTypetag("")) {
      println("$ I was asked to render the background");
      DO_RENDER_BACKGROUND = true;
      return;
    }
  }


  if (theOscMessage.checkAddrPattern("/addPoint")==true) {
    if (theOscMessage.checkTypetag("ffisff")) {
      println("$ I received a point");
      float sentx = theOscMessage.get(0).floatValue();
      float senty = theOscMessage.get(1).floatValue();
      int sentlabel = theOscMessage.get(2).intValue();
      String sentlabelName = theOscMessage.get(3).toString();
      float gain = theOscMessage.get(4).floatValue();
      float tone = theOscMessage.get(5).floatValue();

      pedalPoints.add(sentx, senty, sentlabel,sentlabelName, gain, tone);

      return;
    }
  }
  println("### received an osc message. with address pattern "+theOscMessage.addrPattern());
}
