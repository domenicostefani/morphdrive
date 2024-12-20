PImage backgroundImg;
PSNColors psnColors;
PedalPoints pedalPoints;
ArrayList<PVector> centers;
float SCALE; // Gui scale, from [0,1] coordinates to [width,height] pixel coordinates to display
PVector mousePos;

void setup() {
  size(640, 640);
  SCALE = min(width,height); // Set scale to minimum window dimension
  frameRate(60);
  
  pedalPoints = new PedalPoints();
  psnColors = new PSNColors();
  
  pedalPoints.test_randomInit(50); // generate 50 test points spanning 4 classes
 
  centers = pedalPoints.computeCenters();
  
  /**
   * Render Background once
   */
  drawBackground(SCALE); // Draw Background with (1) measurement points (2) faded color area blobs (3) area barycenters
  save("back.png"); // Save as an image to save computations later in draw()
  backgroundImg = loadImage("back.png");  // Load into backgroundImg
  
  mousePos = new PVector(); // Save mouse position
}

void draw() {
  // Draw rendered background
  image(backgroundImg, 0, 0);
  fill(0);
  stroke(255);
  
  // Allow locking cursor position with mouse press
  if (!cursorLock){
    mousePos.x = mouseX;  
    mousePos.y = mouseY;
    noCursor();
  } else {
    cursor(CROSS);
  }
  
  // Draw pot-shaped cursor
  PVector scalecMousePos = mousePos.copy().div(SCALE);
  float[] closestAreaIdxDist = getClosestCenter(scalecMousePos);
  int closestAreaIdx = int(closestAreaIdxDist[0]);
  float closestAreaDist = closestAreaIdxDist[1];
  float alpha = map(closestAreaDist,0,0.5,255,0);
  color potColor = psnColors.getOpaque(closestAreaIdx,alpha);
  int potRadius = 15;
  // Actual drawing circle
  fill(potColor);
  stroke(0);
  ellipse(mousePos.x, mousePos.y,potRadius*2,potRadius*2);
  // Draw pot intex pointing to closest area
  PVector directionVector = PVector.sub(centers.get(closestAreaIdx), scalecMousePos).normalize();;
  PVector intersectionPoint = PVector.add(mousePos, PVector.mult(directionVector, potRadius));
  // Actual pot-index drawing
  line(mousePos.x,mousePos.y,intersectionPoint.x,intersectionPoint.y);
}

void mousePressed() {
  // Lock cursor position when pressing
  cursorLock = !cursorLock;
}
