
class PSNColors {
  color[] colorLabels = new color[10];
  
  PSNColors() {
    //colorLabels[0] = #ff0000; // red
    //colorLabels[1] = #ff9b00; // orange
    //colorLabels[2] = #0000ff; // blue
    //colorLabels[3] = #ff448f; // pink
    
    colorLabels[0] = #1f77b4; // matplotlib's tab10 tab:blue
    colorLabels[1] = #ff7f0e; // matplotlib's tab10 tab:orange
    colorLabels[2] = #d62728; // matplotlib's tab10 tab:red
    colorLabels[3] = #e377c2; // matplotlib's tab10 tab:pink
    colorLabels[4] = #7f7f7f; // matplotlib's tab10 tab:gray
    colorLabels[5] = #bcbd22; // matplotlib's tab10 tab:olive
    colorLabels[6] = #17becf; // matplotlib's tab10 tab:cyan
    colorLabels[7] = #2ca02c; // matplotlib's tab10 tab:green
    colorLabels[8] = #8c564b; // matplotlib's tab10 tab:brown
    colorLabels[9] = #9467bd; // matplotlib's tab10 tab:purple
  }
  
  color setAlpha(color inColor, int alpha) {
    color currentColor = color(red(inColor),green(inColor),blue(inColor),alpha);
    return currentColor;
  }

  
  color getSolid(int idx){
    if (idx>=0){
      return colorLabels[idx%10];
    }
    return color(0,0,0);
  }
  
  color getOpaque(int idx, float alpha) {
    if (idx<0){
      return color(0,0,0);
    }
    
    color res = colorLabels[idx%10];
    return setAlpha(res,(int)alpha);
  }
}
