/*
    gSystem->AddIncludePath("$AliPhysics_SRC/PWGPP/")
gSystem->AddIncludePath("$AliRoot_SRC/STAT/")
  .L $AliPhysics_SRC/PWGPP/AliRootInteractive.h
  .L ~/github/RootInteractiveTest/JIRA/PWGPP-485/dEdxPerformance.C
   dEdxPerformance(0)

*/

//#include "AliRootInteractive.h"
#include "TString.h"
#include "TTree.h"
#include "TFile.h"

TTree * tree = nullptr;
TTree * trees[6];
 AliTreePlayer player; //dumy load of class to trigger TStatToolkit loading - to avoid bug in ROOT
const char  * inputData[3]={"https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2015/LHC15o/pass1/40MeV_width/Params.root", \
"https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18q/pass1/40MeV_width/Params.root", \
"https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18r/pass1/40MeV_width/Params.root"};

TString options = "tooltips=[('pt', '(@pt)'), ('etaMin', '(@etaMin)'), ('centMin', '(@centMin)'),('pimax', '(@pimax)'), ('corr', '(@corr)'),"
                    "('setting', '(@setting)'),('sizeEl', '(@sizeEl)') ] ,"
                    "layout= '((0, 1),(2, plot_height=100,x_visible=1),commonX=0, x_visible=1,y_visible=1,plot_height=350,plot_width=1000)'";


void InitTrees() {
  TFile::SetCacheFileDir("../data");
  for (Int_t i=0; i<3;  i++){
    TFile * f = TFile::Open(inputData[i],"cacheread");
    trees[i]=(TTree*)f->Get("params");
    trees[i]->SetAlias("p","pt");
    trees[i]->SetAlias("sizePr","(prsigma/prmax<0.15&&prsigma>2.5&&prmax>40)*5");
    trees[i]->SetAlias("sizeEl","(elmeanErr<1&&elmax>70)*5");
    TStatToolkit::AddMetadata(trees[i],"p.AxisTitle","p (GeV/c)");
    TStatToolkit::AddMetadata(trees[i],"elmax.AxisTitle","dEdx_el (a.u.)");
    TStatToolkit::AddMetadata(trees[i],"pimax.AxisTitle","dEdx_pion (a.u.)");
    TStatToolkit::AddMetadata(trees[i],"prmax.AxisTitle","dEdx_proton (a.u.)");
    TStatToolkit::AddMetadata(trees[i],"etaMin.AxisTitle","eta");
  }
}

void dEdxPerformanceAll(const char *html){
  TString query = "centMin<21&pisigma<10&elmax>50";
  TString figureArray= "["
                        "[['p'], ['elmax'], {\"size\": 'sizeEl', 'colorZvar':'setting'}],"
                        "[['p'], ['prmax'], {\"size\": 'sizePr', 'colorZvar':'setting'}],"
                        "[['p'], ['pimax'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "['table'] ]";
  TString optionsAll = "tooltips=[('pt', '(@pt)'), ('etaMin', '(@etaMin)'), ('centMin', '(@centMin)'),('pimax', '(@pimax)'), ('corr', '(@corr)'),"
                    "('setting', '(@setting)'),('sizeEl', '(@sizeEl)'), ('sizePr', '(@sizePr)')] ,"
                    "layout= '((0, 1,2 ),(3, plot_height=100,x_visible=1),commonX=0, x_visible=1,y_visible=1,plot_height=350,plot_width=1000)'";
  TString widgets="query.xx(),slider.p(0,2,0.04,0.12,2),slider.etaMin(0,0.7,0.1,0.0), slider.centMin(0,20,5,5), slider.pimax(40,120,1,40,120),checkbox.corr(), multiselect.setting(0,1,2,3,4,5,6),";
  AliRootInteractive::treeBokehDrawArray("tree", query, figureArray, widgets, optionsAll,html);
}


void dEdxPerformance2(const char *html){
  TString query = "centMin<21&pisigma<10";
  TString figureArray= "["
                        "[['p'], ['pisigma'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "[['p'], ['pimax'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "['table'] ]";
  TString widgets="query.xx(),slider.p(0,2,0.04,0.12,2),slider.etaMin(0,0.7,0.1,0.0), slider.centMin(0,20,5,5), slider.pimax(40,120,1,40,120),checkbox.corr(), multiselect.setting(0,1,2,3,4,5,6),";
  AliRootInteractive::treeBokehDrawArray("tree", query, figureArray, widgets, options,html);
}


void dEdxPerformanceEta(const char *html){
  TString query = "centMin<21&pisigma<10";
  TString figureArray= "["
                        "[['etaMin'], ['pisigma'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "[['etaMin'], ['pimax'], {\"size\": 5, 'colorZvar':'setting'}],"
                        "['table'] ]";
  TString widgets="query.xx(),slider.p(0,2,0.04,1.02),slider.etaMin(0,0.7,0.1,0.,0.8), slider.centMin(0,20,5,5), slider.pimax(40,120,1,40,120),checkbox.corr(), multiselect.setting(0,1,2,3,4,5,6),";
  AliRootInteractive::treeBokehDrawArray("tree", query, figureArray, widgets, options,html);
}



void dEdxPerformance(Int_t i=0){
  if (tree== nullptr) InitTrees();
  tree=trees[i];
    dEdxPerformanceAll(Form("dedxPtElPi_%d.html",i));
    dEdxPerformance2(Form("dedxPtResPi_%d.html",i));
    dEdxPerformanceEta(Form("dedxEta_%d.html",i));

}
