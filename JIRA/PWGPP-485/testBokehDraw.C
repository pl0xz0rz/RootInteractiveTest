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
  x+="makeAliasAnyTree(\"MIPquality_Warning\",base,aliases)\n";
  x+="print(RenderTree(base))";
  TPython::Exec(x);
}

void treeBokehDraw(const char * treeName ,const char *varX, const char *vary, const char *varz, const char *layout, const char *option){
  TString x=importBokeh;
  x+="tree=ROOT.gROOT.GetGlobal(\"tree\") \n";
  x+=TString::Format("");
}


void testEvalMatrix(Int_t m, Bool_t doPrint, Int_t nLoops=1000){
  testMatrix=new TMatrixD(m,m);
  for (Int_t i=0;i<m; i++)
    for (Int_t j=0;j<m; j++)
      (*testMatrix)(i,j)=gRandom->Rndm();
  TString sumP=R"raw(m=ROOT.gROOT.GetGlobal("testMatrix")\n)raw";
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
