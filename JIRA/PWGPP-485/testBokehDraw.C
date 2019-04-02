/*
 .L $HOME/github/RootInteractiveTest/JIRA/PWGPP-485/testBokehDraw.C
 */
TString importBokeh="";
TTree * tree=0;
void InitBokeh() {
  importBokeh  = "from Tools.aliTreePlayer import * \n";
  importBokeh += "from anytree import * \n";
  importBokeh += "from InteractiveDrawing.bokeh.bokehDraw import *\n";
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