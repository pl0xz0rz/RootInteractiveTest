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
  importBokeh += "from InteractiveDrawing.bokeh.bokehDrawSA import *\n";
  importBokeh += "import numpy as np\n";
}

TTree *  makeABCtree(Int_t nPoints){
    TTreeSRedirector *pcstream = new TTreeSRedirector("treeABCD.root","recreate");
    Double_t abcd[4];
    Bool_t isABig;
    Int_t E;
    for (Int_t i=0; i<nPoints; i++){
        for (Int_t j=0; j<4; j++) abcd[j]=gRandom->Rndm();
        isABig = (abcd[0]>0.5);
        E = (abcd[1]/0.2);
        (*pcstream)<<"tree"<<
                   "A="<<abcd[0]<<
                   "B="<<abcd[1]<<
                   "C="<<abcd[2]<<
                   "D="<<abcd[3]<<
                   "E="<<E<<
                   "Bool="<<isABig<<
                   "\n";
    }
    delete pcstream;
    TFile *f = TFile::Open("treeABCD.root");
    return (TTree*)f->Get("tree");
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

void treeBokehDraw(const char * treeName ,const char *varX, const char *varY, const char *varZ, const char *layout, const char *option){
  TString x=importBokeh;
  x+="tree=ROOT.gROOT.GetGlobal(\"tree\") \n";
  x+=TString::Format("");
}

void treeBokehDrawArray(const char * treeName, const char *query, const char *figureArray,  const char *widgets, const char *options){
    initPython();
    TString x=importBokeh;
    x = x + "tree=ROOT.gROOT.GetGlobal(\""+treeName+"\")\n" +
            "fig=bokehDrawSA.fromArray(tree ,'" + query + "'," + figureArray + ",'" + widgets + "'," + options+")";
//    std::cout<<x<<std::endl;
    TPython::Exec(x);
}

void testBokehDrawArray(){
    tree= makeABCtree(1000);
    TString query = "A>0";
    TString figureArray= "["
                         "[['A'], ['D+A','C-A'], {\"size\": 1}],"
                         "[['A'], ['C+A', 'C-A']],"
                         "[['A'], ['C']]]";
    TString widgets="slider.A(0,1,0.01,0.1,0.9),slider.B(0,1,0.01,0.1,0.9),"
                    "slider.C(0,1,0.01,0.1,0.9),slider.D(0,1,0.01,0.1,0.9)";
    TString options = "tooltips=[('VarA', '(@A)'), ('VarB', '(@B)'), ('VarC', '(@C)'), ('VarD', '(@D)')],"
                      "layout= '((0,1),(2, x_visible=1),commonX=1, x_visible=1,y_visible=0,plot_height=250,plot_width=1000)'";

    treeBokehDrawArray("tree", query, figureArray, widgets, options);
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

