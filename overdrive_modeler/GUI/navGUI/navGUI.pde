import oscP5.*;
import netP5.*;

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

void setup() {
  size(640, 640);
  SCALE = min(width, height); // Set scale to minimum window dimension
  frameRate(60);

  // Initialize OSC
  oscP5 = new OscP5(this, 12000); // Listen for incoming OSC messages on port 12000 (if needed)
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
}

boolean DO_RENDER_BACKGROUND = false;

void renderBackground() {
  if (!rendering) {
    rendering = true;

    centers = pedalPoints.computeCenters();
    drawBackground(SCALE); // Draw Background with (1) measurement points (2) faded color area blobs (3) area barycenters
 //<>// //<>//
    save("back.png"); // Save as an image to save computations later in draw()
    backgroundImg = loadImage("back.png");  // Load into backgroundImg

    rendering = false;
  }
}

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

  // Allow locking cursor position with mouse press
  if (!cursorLock) {
    mousePos.x = mouseX;
    mousePos.y = mouseY;
    noCursor();
  } else {
    cursor(CROSS);
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
    if (theOscMessage.checkTypetag("ffis")) {
      println("$ I received a point");
      float sentx = theOscMessage.get(0).floatValue();
      float senty = theOscMessage.get(1).floatValue();
      int sentlabel = theOscMessage.get(2).intValue();
      String sentlabelName = theOscMessage.get(3).toString();

      pedalPoints.add(sentx, senty, sentlabel,sentlabelName);

      return;
    }
  }
  println("### received an osc message. with address pattern "+theOscMessage.addrPattern());
}
