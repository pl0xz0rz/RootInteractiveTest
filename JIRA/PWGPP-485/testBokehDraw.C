/*
 .L $HOME/github/RootInteractiveTest/JIRA/PWGPP-485/testBokehDraw.C

 initPython()

 */
TString importBokeh="";
TTree * tree=0;
TMatrixD *testMatrix=0;
TStopwatch timer;

void initPython() {
  importBokeh  = "from Tools.aliTreePlayer import * \n";
  importBokeh += "from anytree import * \n";
  importBokeh += "from InteractiveDrawing.bokeh.bokehDraw import *\n";
  importBokeh += "import numpy as np\n";
}

void InitData(){
  AliExternalInfo info;
  tree=info.GetTree("QA.TPC","LHC15o","cpass1_pass1");
}

void testBokehRender(){
  TString x=importBokeh;
  x+="tree=ROOT.gROOT.GetGlobal(\"tree\") \n";
  x+="aliases=aliasToDictionary(tree)\n";
  x+="base=Node(\"MIPquality_Warning\") \n";
  x+="makeAliasAnyTree(\"MIPquality_Warning\",aliases,base)\n";
  x+="print(RenderTree(base))";
  TPython::Exec(x);
}

void treeBokehDraw(const char * treeName ,const char *varX, const char *vary, const char *varz, const char *layout, const char *option){
  TString x=importBokeh;
  x+="tree=ROOT.gROOT.GetGlobal(\"tree\") \n";
  x+=TString::Format("");
}

void treeBokehDrawArray(const char *treeName, const char *figureArray,  const char *widgets, const char *options){
    TString x=importBokeh;
    x+="fig=bokehDraw.fromArray(";
    x+=treeName;
    x+=", \"1\",";
    x+=figureArray;
    x+=",";
    x+=widgets;
    x+=",";
    x+=options;
    x+=")";
    std::cout<<x<<std::endl;
    TPython::Exec(x);
}

void testBokehDrawArray(){
    initPython();
    InitData();
    testBokehRender();
    TString figureArray="["
                    "[['chunkMedian'], ['meanMIP','meanMIPele'], {\"color\": \"red\", \"size\": 2, \"colorZvar\":\"MIPquality_Warning\"}],"
                    "[['fraction'], ['resolutionMIP'], {\"color\": \"red\", \"size\": 2, \"colorZvar\":\"MIPquality_Warning\"}],"
                    "['table']"
                    "]";
    TString widgets="\"tab.sliders(slider.meanMIP(45,55,0.1,45,55),slider.meanMIPele(50,80,0.2,50,80),"
                    "slider.resolutionMIP(0,0.15,0.01,0,0.15)),tab.checkboxGlobal(slider.global_Warning(0,1,1,0,1),"
                    "checkbox.global_Outlier(0)),tab.checkboxMIP(slider.MIPquality_Warning(0,1,1,0,1),"
                    "checkbox.MIPquality_Outlier(0), checkbox.MIPquality_PhysAcc(1))\"";
    TString options = "tooltips=tooltips, layout=figureLayout,nEntries=100000000,x_axis_type='datetime'";
    TString init ="tooltips=[(\"MIP\",\"(@meanMIP)\"),(\"Electron\",\"@meanMIPele\"),"
    "(\"Global status\",\"(@global_Outlier,@global_Warning)\"),"
    "(\"MIP status(Warning,Outlier,Acc.)\",\"@MIPquality_Warning,@MIPquality_Outlier,@MIPquality_PhysAcc\")]\n"
    "figureLayout: str = '((0,1),(2, x_visible=1),commonY=1, x_visible=1,y_visible=0,plot_height=250,plot_width=1000)'";
    std::cout<<init<<std::endl;
    TPython::Exec(init);
    treeBokehDrawArray("tree", figureArray, widgets, options);
}

void testEvalMatrix(Int_t m, Bool_t doPrint, Int_t nLoops=1000){
  testMatrix=new TMatrixD(m,m);
  for (Int_t i=0;i<m; i++)
    for (Int_t j=0;j<m; j++)
      (*testMatrix)(i,j)=gRandom->Rndm();
  TString sumP=R"raw(m=ROOT.gROOT.GetGlobal("testMatrix");)raw";
  if (doPrint){
    sumP += "print(m.Determinant())";
  }else {
    sumP += "m.Determinant()";
  }
  printf("Python loop\n");
  timer.Start(); for (Int_t i=0; i<nLoops;i++)  {(*testMatrix)(0,0)+=1;TPython::Exec(sumP);} timer.Print();

  printf("C loop\n");
  timer.Start(); for (Int_t i=0; i<nLoops;i++)  {(*testMatrix)(0,0)+=1;testMatrix->Determinant();} timer.Print();
}
