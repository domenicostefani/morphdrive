void drawBackground(float SCALE){
  //background(200);
  background(240,240,255);
  
  for (int i=0; i<centers.size(); ++i){
    float radius = computeRadiusPerLabel(pedalPoints.getAll(), i, centers.get(i));
    
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
    drawXMark(centers.get(i).x*SCALE,centers.get(i).y*SCALE,10);
  }
}
