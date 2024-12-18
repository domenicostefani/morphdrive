
color[] colorLabels = new color[10];
PImage img;



ArrayList<PedalPoint> points;
ArrayList<PVector> centers;

float SCALE = 1;
PVector mousePos;

void setup() {
  size(640, 640);
  SCALE = min(width,height);
  
  //frameRate(1);
  
  points = new ArrayList<PedalPoint>();
  centers = new  ArrayList<PVector>();
  
  
  colorLabels[0] = color(255,0,0);
  colorLabels[1] = color(255,155,0);
  colorLabels[2] = color(0,0,255);
  colorLabels[3] = color(255,68,143);
  
  for (int i=0; i<50; ++i){
    float x, y;
    x = random(1);
    y = random(1);
    int label = x< 0.5 && y<0.5 ? 0: (x>= 0.5 && y<0.5)?1:(x<0.5 ? 2:3);
    
    
    PedalPoint toadd = new PedalPoint(x,y, label);
    toadd.draw_scale = SCALE;
    
    points.add(toadd);
    
  }
  
  for (int lidx=0; lidx<4; ++lidx){
    PVector point = computeCentersPerLabel(points,lidx);
    centers.add(point);
  }
  
  
  drawBackground();
  save("back.png");
  
  img = loadImage("back.png");
  
  mousePos = new PVector();
}


void draw() {
  image(img, 0, 0);
  fill(0);
  stroke(255);
  if (!cursorLock){
    mousePos.x = mouseX;
    mousePos.y = mouseY;
  }
  
  PVector scalecMousePos = mousePos.copy();
  float[] colIdxDist = getClosestCenter(scalecMousePos.div(SCALE));
  int colIdx = int(colIdxDist[0]);
  float centerDist = colIdxDist[1];
  
  println("colIdx: ",colIdx, " dist ",centerDist);
  
  float alpha = 255;
  color currentColor = color(red(colorLabels[colIdx]),green(colorLabels[colIdx]),blue(colorLabels[colIdx]),alpha);
  fill(currentColor);
  ellipse(mousePos.x, mousePos.y,30,30);
  line(mousePos.x,mousePos.y,mousePos.x,mousePos.y-15);
}

void mousePressed() {
  cursorLock = !cursorLock;
}
