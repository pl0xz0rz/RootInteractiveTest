/*
  .L $AliPhysics_SRC/PWGPP/AliRootInteractive.h
  .L ~/github/RootInteractiveTest/JIRA/PWGPP-485/dEdxPerformance.C
   dEdxPerformance(0)

*/

TTree * tree = nullptr;
TTree * trees[6];
const char  * inputData[3]={"https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2015/LHC15o/pass1/Params.root", \
  "https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18q/pass1New/Params.root",  \
        "https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18r/pass1New/Params.root"};


void InitTrees() {
  TFile::SetCacheFileDir("../data");
  for (Int_t i=0; i<3;  i++){
    TFile * f = TFile::Open(inputData[i],"cacheread");
    trees[i]=(TTree*)f->Get("params");
  }
}

void dEdxPerformance1(){
  tree->SetAlias("sizeEl","(elmeanErr<1&&elmax>70)*5");
  TString query = "centMin<21&setting<5&elmax>pimax";
  TString figureArray= "["
                        "[['pt'], ['elmax'], {\"size\": 'sizeEl', 'colorZvar':'setting'}],"
                        "[['pt'], ['pimax'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "['table'] ]";
  TString widgets="query.xx(),slider.pt(0,2,0.04,0.1,2),slider.etaMin(0,0.7,0.1,0.7), slider.centMin(0,20,5,10), slider.pimax(40,120,1,40,120),slider.corr(0,1,1,0), slider.setting(0,6,1,0,6),";
  TString options = "tooltips=[('pt', '(@pt)'), ('etaMin', '(@etaMin)'), ('centMin', '(@centMin)'),('pimax', '(@pimax)'), ('corr', '(@corr)'), ('setting', '(@setting)'),('sizeEl', '(@sizeEl)') ] ,"
                      "layout= '((0, 1),(2, plot_height=100,x_visible=1),commonX=0, x_visible=1,y_visible=1,plot_height=450,plot_width=1000)'";
  AliRootInteractive::treeBokehDrawArray("tree", query, figureArray, widgets, options);
}


void dEdxPerformance2(){
  TString query = "centMin<21&elmax>50&setting<5&pisigma<10";
  TString figureArray= "["
                        "[['pt'], ['pisigma'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "[['pt'], ['pimax'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "['table'] ]";
  TString widgets="query.xx(),slider.pt(0,2,0.04,0.1,2),slider.etaMin(0,0.7,0.1,0.7), slider.centMin(0,20,5,10), slider.pimax(40,120,1,40,120),slider.corr(0,1,1,0), slider.setting(0,6,1,0,6),";
  TString options = "tooltips=[('pt', '(@pt)'), ('etaMin', '(@etaMin)'), ('centMin', '(@centMin)'),('pimax', '(@pimax)'), ('corr', '(@corr)'), ('setting', '(@setting)')],"
                      "layout= '((0, 1),(2, plot_height=100,x_visible=1),commonX=0, x_visible=1,y_visible=1,plot_height=450,plot_width=1000)'";
  AliRootInteractive::treeBokehDrawArray("tree", query, figureArray, widgets, options);
}


void dEdxPerformanceEta(){
  tree->SetAlias("sizeEl","(elmeanErr<1)*2");
  TString query = "centMin<21&elmax>50&setting<5&pisigma<10";
  TString figureArray= "["
                        "[['etaMin'], ['pisigma'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "[['etaMin'], ['pimax'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "['table'] ]";
  TString widgets="query.xx(),slider.pt(0,2,0.02,1),slider.etaMin(0,0.7,0.1,0.,0.8), slider.centMin(0,20,5,10), slider.pimax(40,120,1,40,120),slider.corr(0,1,1,0), slider.setting(0,6,1,0,6),";
  TString options = "tooltips=[('pt', '(@pt)'), ('etaMin', '(@etaMin)'), ('centMin', '(@centMin)'),('pimax', '(@pimax)'), ('corr', '(@corr)'), ('setting', '(@setting)')],"
                      "layout= '((0, 1),(2, plot_height=100,x_visible=1),commonX=0, x_visible=1,y_visible=1,plot_height=450,plot_width=1000)'";
  AliRootInteractive::treeBokehDrawArray("tree", query, figureArray, widgets, options);
}



void dEdxPerformance(Int_t i=0){
  if (tree== nullptr) InitTrees();
  tree=trees[i];
  dEdxPerformance1();
  dEdxPerformance2();
  dEdxPerformanceEta();
}