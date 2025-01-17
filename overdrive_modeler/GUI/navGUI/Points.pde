import java.util.Collections;

class PedalPoint {
  PVector position;
  int label;
  float gain = 0;
  float tone = 0;

  
  PedalPoint(float x, float y, int label, float gain, float tone){
    position = new PVector(x, y);
    this.label = label;
    this.gain = gain;
    this.tone = tone;
  }
  
  PedalPoint(PVector position, int label, float gain, float tone){
    this.position = position;
    this.label = label;
    this.gain = gain;
    this.tone = tone;
  }
  
  void draw(float draw_scale){
    fill(psnColors.getSolid(label));
    stroke(psnColors.getSolid(label));
    
    pushMatrix();
    ellipse(position.x*draw_scale,position.y*draw_scale,10,10);
    popMatrix();
  }

  PVector getPosition(){
    return position;
  }

  float getX(){
    return position.x;
  }

  float getY(){
      return position.y;
  }

  float getGain(){
    return gain;
  }

  float getTone(){
    return tone;
  }
  
}



PVector calculateBarycenter(ArrayList<PVector> points) {
  // If the list is empty, return null
  if (points == null || points.isEmpty()) {
    return null;
  }
  
  // Initialize variables to store sum of x, y coordinates
  float sumX = 0;
  float sumY = 0;
  
  // Sum up all coordinates
  for (PVector point : points) {
    sumX += point.x;
    sumY += point.y;
  }
    
  // Calculate the average (barycenter)
  int pointCount = points.size();
  PVector point = new PVector(
    sumX / pointCount, 
    sumY / pointCount, 
    0.0
  );
    
  return point;
}

PVector computeCentersPerLabel(ArrayList<PedalPoint> points, int label) {
  
  
  ArrayList<PVector> relevantPoints = new ArrayList<PVector>();
  for(int i=0; i<points.size(); ++i){
    if (points.get(i).label == label){
      relevantPoints.add(points.get(i).position);
    }
  }
  
  PVector avg = calculateBarycenter(relevantPoints);
  return avg;
}

float computeRadiusPerLabel(ArrayList<PedalPoint> points, int label, PVector center){
  float maxDistance = 0;
  for(int i=0; i<points.size(); ++i){
    float dist = center.dist(points.get(i).position);
    if (points.get(i).label == label && dist > maxDistance){
      maxDistance = dist;
    }
  }
  return maxDistance;
}

void drawXMark(float x, float y, float size) {
  // Set line properties
  strokeWeight(2);
  
  // Draw the X
  line(x - size/2, y - size/2, x + size/2, y + size/2);
  line(x - size/2, y + size/2, x + size/2, y - size/2);
}





boolean cursorLock = false;

float[] getClosestCenter(PVector pos){
  int closestIdx = -1;
  float mindist = 10000000;
  for (int i=0;i<centers.size();++i){
    float curdist = pos.dist(centers.get(i));
    //println("Centerceck ", i, "[",centers.get(i).x,",",centers.get(i).y,"]"," curdist: ", curdist);
    if (curdist < mindist) {
      mindist = curdist;
      closestIdx = i;
    }
  }
  float[] res = new float[2];
  res[0] = float(closestIdx);
  res[1] = mindist;
  return res;
}

/**
 * Class with list of points representing different pedals in the 2D representation of the latent space
 */
class PedalPoints {
  ArrayList<PedalPoint> points;
  ArrayList<Integer> classes;
  ArrayList<String> classnames;
  ArrayList<PVector> centersCache;
  
  PedalPoints(){
    points = new ArrayList<PedalPoint>();
    classes = new ArrayList<Integer>();
    classnames = new ArrayList<String>();
    centersCache = new ArrayList<PVector>();
  }
  
  
  int size(){
    return points.size();
  }
  
  void add(float x, float y, int pClass, String pClassName, float gain, float tone){
    PedalPoint toadd = new PedalPoint(x,y, pClass, gain, tone); //TODO: fix gaintone
    points.add(toadd);

    boolean inclasses = false;
    for (int ci=0; ci<classes.size(); ++ci) {
      if (classes.get(ci) == pClass) {
        inclasses = true;
        break;
      }
    }
    if (!inclasses) {
      classes.add(pClass);
      classnames.add(pClassName);
    }  
  }
  
  /**
   * Random initialization JUST FOR TEST
   */
  void test_randomInit(int numPedals) {
    classes.clear(); points.clear();
    for (int i=0; i<numPedals; ++i){
      float x = random(1), y = random(1);
      int pClass = x< 0.5 && y<0.5 ? 0: (x>= 0.5 && y<0.5)?1:(x<0.5 ? 2:3); // Choose pedal class depending on space corner
      String classname = "test "+pClass;
      this.add(x,y,pClass,classname, x,y);
    }
  }

  void clear(){
    points.clear();
    classes.clear();
    classnames.clear();
  }

  String getClassname(int idx){
    if (idx < 0 || idx >= classnames.size()) {
      println("Bad index ",idx);
      return null;
    }
    println("Returning ",classnames.get(idx));
    return classnames.get(idx);
  }
  
  PedalPoint get(int index) {
    if (index <0 || index >= points.size()) {
      println("Bad index ",index);
      return null;
    }
    return points.get(index);
  }
  
  ArrayList<PVector> computeCenters(){
    centersCache.clear();
    println("Computing centers");
    
    for (int cidx=0; cidx<classes.size(); ++cidx){
      int classValue = classes.get(cidx);
      PVector point = computeCentersPerLabel(points,classValue);
      centersCache.add(point);
    }
    return centersCache;
  }
  ArrayList<PVector> getCenters(){ return centersCache; }
  
  void draw(int index, float scale){
    if (index <0 || index >= points.size()) {
      println("Bad index ",index);
      return;
    }
    points.get(index).draw(scale);
  }
  
  void drawAll(float scale){
    for (int i=0; i<points.size(); ++i){
      points.get(i).draw(scale);
    }
  }
  
  ArrayList<PedalPoint> getAll(){
    ArrayList<PedalPoint> res = new ArrayList<PedalPoint>();
    for (int i=0; i<points.size(); ++i){
      res.add(points.get(i));
    }
    return res;
  }

  PedalPoint getHovering(PVector pos){
    for (int i=0; i<points.size(); ++i){
      if (pos.dist(points.get(i).position) < 0.01){
        return points.get(i);
      }
    }
    return null;
  }
  
}
