void drawBackground(float SCALE){
  //background(200);
  background(240,240,255);
  println("About to draw background");
  
  ArrayList<PedalPoint> all_the_points = pedalPoints.getAll();

  
  println("Points: ",all_the_points.size());
  println("Centers to draw: ",centers.size());
  
  for (int i=0; i<centers.size(); ++i){
    float radius = computeRadiusPerLabel(all_the_points, i, centers.get(i));
    
    int maxDivs = 125;
    for (int sd = 0; sd<maxDivs; sd++){ 
      float alpha = 255/maxDivs;
      color currentColor = psnColors.getOpaque(i,alpha);
      fill(currentColor);
      noStroke();
      float diam = map(sd,0,maxDivs,radius*SCALE*2,0);
      ellipse(centers.get(i).x*SCALE,centers.get(i).y*SCALE,diam,diam);
    }
  }
  
  pedalPoints.drawAll(SCALE);
  
  for (int i=0; i<centers.size(); ++i){
    fill(255,255,255);
    stroke(255,255,255);
    println("Drawing center ",i," at ",centers.get(i).x*SCALE,",",centers.get(i).y*SCALE);
    float centerx = centers.get(i).x*SCALE;
    float centery = centers.get(i).y*SCALE;
    drawXMark(centerx,centery,10);
    // Draw text at centerx+10,centery+10 with label string
    fill(255);
    stroke(255);
    strokeWeight(3);
    textSize(20);
    String label = pedalPoints.getClassname(i);
    float lwidth = textWidth(label);

    int offsetYsign = -1;
    int offsetXsign = 1;
    int offsetX = 15;
    int offsetY = 15;

    if (centerx+offsetX*offsetXsign+lwidth > width){
      offsetXsign = -1;
      offsetX = int(lwidth);
    }

    text(label,centerx+offsetX*offsetXsign,centery+offsetY*offsetYsign);
  }
}
