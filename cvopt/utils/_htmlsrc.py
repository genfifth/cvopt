pjs_setting = {"hyperopt"  :{"AGENT_TYPE":'"randomwalk"', "CIRCLE_TYPE":'"line"', "min_n_circle":1000, "max_n_circle":4000}, 
               "gaopt"      :{"AGENT_TYPE":'"ga"', "CIRCLE_TYPE":'"normal_sw"', "min_n_circle":500, "max_n_circle":3000}, 
               "bayesopt"   :{"AGENT_TYPE":'"randomwalk"', "CIRCLE_TYPE":'"quad"', "min_n_circle":1000, "max_n_circle":3000}, 
               "randomopt" :{"AGENT_TYPE":'"random"', "CIRCLE_TYPE":'"normal"', "min_n_circle":50, "max_n_circle":500}, 
              }
n_iter_setting = {"min":0, "max":256}


additional_head = """
<style>
  .bk-root {width:1000px; margin-left:auto; margin-right:auto;}
  #bg { position:fixed;    top:0; left:0; width:100%; height:100%; }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/processing.js/1.6.6/processing.js"></script>
"""

pjs = """
<script type="application/processing">
 


class Circle{
  float x, y;
  float radius;
  color linecolor;
  //color fillcolor;
  float alpha;
  String type;
  
  Circle(float init_x, float init_y, float init_radius, 
         float init_alpha, color init_linecolor, String init_type) { 
    x = init_x;
    y = init_y;
    radius = init_radius;
    alpha = init_alpha;
    linecolor = init_linecolor;
    //fillcolor = init_fillcolor;
    type = init_type;
  }
  void show() {
    stroke(linecolor, alpha);
    if (type == "normal"){
      noFill();
      ellipse(x, y, radius*2, radius*2);
    }
    if (type == "normal_sw"){
      noFill();
      strokeWeight(random(0.5,5));
      ellipse(x, y, radius*2, radius*2);
    }
    else if (type == "line"){
      float angle_step = 25;
      for (float angle=0; angle <= 360*2; angle+=angle_step) {
        float rad = radians(angle);
        float rad_next = radians(angle+angle_step*random(0.8,1.2));
        strokeWeight(random(1,2));
        line(x+radius*cos(rad), y+radius*sin(rad), 
             x+radius*cos(rad_next)+rad*random(1), 
             y+radius*sin(rad_next)+rad*random(1));
        
        if (angle > random(360, 720)){
          break;
        }
      }
    }
    else if (type == "quad"){
      float angle_step = 17;
      beginShape(QUAD_STRIP);
      for (float angle=0; angle <= 360; angle+=angle_step) {
        if (0.8 > random(1)){
          fill(#ffffff, random(0,50));
          stroke(linecolor, alpha);
        }
        else{
          noFill();
          noStroke();
        }
        
        float rad = radians(angle);
        vertex(x+radius*cos(rad)+rad*random(1), y+radius*sin(rad)+rad*random(1));

      }
      endShape();
    }
  }
}


class Agent{
  int time;
  PVector cr_pos;
  PVector[] poss = {};
  String type;
  float x, y;
  
  float rd_max, rw_seed, rw_step, rad;

  Agent(float init_x, float init_y, String init_type){
    cr_pos = new PVector(init_x, init_y);    
    poss = (PVector[])append(poss, cr_pos);
    time = 0;
    type = init_type;
    
    rd_max = 100;
    rw_seed = 1;
    rw_step = (disp_width+disp_height)/50;
  }
  void step(){
    // algorism
    if (type == "random"){
      x = random(rd_max);
      x = map(x, 0, rd_max, 0, disp_width);
      y = random(rd_max);
      y = map(y, 0, rd_max, 0, disp_height);
    }
    else if (type == "randomwalk"){
      rad = random(rd_max);
      rad = map(rad, 0, 1, 0, 2*PI);
      x = poss[time].x + rw_step*cos(rad);
      y = poss[time].y + rw_step*sin(rad);
    }
    else if (type == "ga"){
      float r = random(1);
      if ((time < 10) || r < 0.05){
        x = random(rd_max);
        x = map(x, 0, rd_max, 0, disp_width);
        y = random(rd_max);
        y = map(y, 0, rd_max, 0, disp_height);
      }
      else{
        int p_0 = int(random(0, time));
        int p_1 = int(random(0, time));
        
        //float w = random(1);
        //x = w*poss[p_0].x + (1-w)*poss[p_1].x;
        //y = w*poss[p_0].y + (1-w)*poss[p_1].y;
        x = poss[p_0].x + poss[p_1].x;
        y = poss[p_0].y + poss[p_1].y;        
      }
    
    }
    // for out of screen
    if (abs(x) > disp_width*1.5){
      x = x * 0.5;
    }
    if (abs(y) > disp_width*1.5){
      y = y * 0.5;
    }
    
    cr_pos = new PVector(x, y);
    poss = (PVector[])append(poss, cr_pos);
    time ++;
  }
}



int N_CIRCLE = REP_N_CIRCLE;
String AGENT_TYPE = REP_AGENT_TYPE;
String CIRCLE_TYPE = REP_CIRCLE_TYPE;

Circle[] circles = {};
Agent agent;
float disp_width, disp_height;

void setup() {  
  size(innerWidth, innerHeight); 
  colorMode(RGB, 255, 255, 255, 100);
  background(#f2f2f2);
  smooth();
  noLoop();
  disp_width = innerWidth;
  disp_height = innerHeight;
  
  agent = new Agent(disp_width/2, disp_height/2, AGENT_TYPE);
}

void draw() {
  for (int i = 0; i < N_CIRCLE; i++) {
    agent.step();
    float radius = random(1, 150);
    //float radius = 300;
    float alpha = 100;
    color linecolor = #b7b7b7;
    Circle circle = new Circle(agent.cr_pos.x, agent.cr_pos.y, 
                               radius, alpha, linecolor, CIRCLE_TYPE);
    circles = (Circle[])append(circles, circle);
  }
  
  for (int i = 0; i < N_CIRCLE; i++) {
    Circle circle = circles[i];
    circle.show();
  }
  
}


        
</script>
<canvas id="bg"></canvas>
"""