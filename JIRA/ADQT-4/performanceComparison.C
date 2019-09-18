/*
    gSystem->AddIncludePath("$AliPhysics_SRC/PWGPP/")
gSystem->AddIncludePath("$AliRoot_SRC/STAT/")
  .L $AliPhysics_SRC/PWGPP/AliRootInteractive.h
  .L ~/github/RootInteractiveTest/JIRA/ADQT-4/performanceComparison.C
  InitTrees();
drawQMaxComparison()
drawChi2ITSComparison()

*/

//#include "AliRootInteractive.h"
#include "TString.h"
#include "TTree.h"
#include "TFile.h"

TTree * treeSec = nullptr;
TTree * trees[6];
AliTreePlayer player; //dumy load of class to trigger TStatToolkit loading - to avoid bug in ROOT

TString options = "tooltips=[('pt', '(@pt)'), ('etaMin', '(@etaMin)'), ('centMin', '(@centMin)'),('pimax', '(@pimax)'), ('corr', '(@corr)'),"
                    "('setting', '(@setting)'),('sizeEl', '(@sizeEl)') ] ,"
                    "layout= '((0, 1),(2, plot_height=100,x_visible=1),commonX=0, x_visible=1,y_visible=1,plot_height=350,plot_width=1000)'";


void InitTrees() {
  TFile::SetCacheFileDir("../data/");
  treeSec=AliTreePlayer::LoadTrees("cat map.list",".*_dSec.*","xxx",".*","","");
  treeSec->SetAlias("qPt","qPtCenter");
  treeSec->SetAlias("dSector","dSectorCenter");
  treeSec->SetAlias("atgl","atglCenter");
  treeSec->SetAlias("multTPCClusterN","multTPCClusterNCenter");
  TStatToolkit::AddMetadata(treeSec,"dSector.AxisTitle","x_sector (a.u.)");
  TStatToolkit::AddMetadata(treeSec,"qPt.AxisTitle","x_sector (a.u.)");
  TStatToolkit::AddMetadata(treeSec,"atgl.AxisTitle","pz/pt");
  TStatToolkit::AddMetadata(treeSec,"multTPCClusterN.AxisTitle","Ncl/Ncl_central");
  //
  treeSec->SetAlias("LHC15oQMax0","LHC15o_pass1.hisQMaxMIP0_dSecDist.binMedian");
  treeSec->SetAlias("LHC18qQMax0","LHC18q_pass1.hisQMaxMIP0_dSecDist.binMedian");
  treeSec->SetAlias("LHC18l8aQMax0","LHC18l8a_MC.hisQMaxMIP0_dSecDist.binMedian");
  //
  //
  treeSec->SetAlias("LHC15oChi2ITS","LHC15o_pass1.hisnormChi2ITS_dSecDist.binMedian");
  treeSec->SetAlias("LHC18qChi2ITS","LHC18q_pass1.hisnormChi2ITS_dSecDist.binMedian");
  treeSec->SetAlias("LHC18l8aChi2ITS","LHC18l8a_MC.hisnormChi2ITS_dSecDist.binMedian");
  //
  treeSec->SetAlias("LHC15oChi2TPC","LHC15o_pass1.hisnormChi2TPC_dSecDist.binMedian");
  treeSec->SetAlias("LHC18qChi2TPC","LHC18q_pass1.hisnormChi2TPC_dSecDist.binMedian");
  treeSec->SetAlias("LHC18l8aChi2TPC","LHC18l8a_MC.hisnormChi2TPC_dSecDist.binMedian");

}

void drawQMaxComparison(){
   TString query = "entries>50";
  TString figureArray= "["
                        "[['dSector'], ['LHC15oQMax0'], {'colorZvar':'qPt'}],"
                        "[['dSector'], ['LHC18qQMax0'], {'colorZvar':'qPt'}],"
                        "[['dSector'], ['LHC18l8aQMax0'], {'colorZvar':'qPt'}],"
                        "['table'] ]";
  TString optionsAll = ""
  "tooltips=[('qPt', '(@qPt)'), ('ps/pt', '(@atgl)'), ('multTPCClusterN', '(@multTPCClusterN)'),"
  "('LHC15oQMax0','(@LHC15oQMax0)'), ('LHC18qQMax0', '(@LHC18qQMax0)'),"
  "('LHC18l8aQMax0', '(@LHC18l8aQMax0)')] ,"
  "layout= '((0, 1,2 ),(3, plot_height=100,x_visible=1),commonX=0, commonY=0,x_visible=1,y_visible=1,plot_height=350,plot_width=1000)'";

  TString widgets="query.xx(),slider.qPt(-3,3,0.3,-0.3,0.3),slider.atgl(0,1,0.1,0.1), slider.dSector(0,1,0.05,0,1),slider.multTPCClusterN(0,1.4,0.2,0.2)";
  AliRootInteractive::treeBokehDrawArray("treeSec", query, figureArray, widgets, optionsAll,"drawQMaxComparison.html");
}


void drawChi2ITSComparison(){
   TString query = "entries>50";
  TString figureArray= "["
                        "[['dSector'], ['LHC15oChi2ITS'], {'colorZvar':'qPt'}],"
                        "[['dSector'], ['LHC18qChi2ITS'], {'colorZvar':'qPt'}],"
                        "[['dSector'], ['LHC18l8aChi2ITS'], {'colorZvar':'qPt'}],"
                        "['table'] ]";
  TString optionsAll = ""
  "tooltips=[('qPt', '(@qPt)'), ('ps/pt', '(@atgl)'), ('multTPCClusterN', '(@multTPCClusterN)'),"
  "('LHC15oChi2ITS','(@LHC15oChi2ITS)'), ('LHC18qChi2ITS', '(@LHC18qChi2ITS)'),"
  "('LHC18l8aChi2ITS', '(@LHC18l8aChi2ITS)')] ,"
  "layout= '((0, 1,2 ),(3, plot_height=100,x_visible=1),commonX=0, commonY=0,x_visible=1,y_visible=1,plot_height=350,plot_width=1000)'";

  TString widgets="query.xx(),slider.qPt(-3,3,0.3,-0.3,0.3),slider.atgl(0,1,0.1,0.1), slider.dSector(0,1,0.05,0,1),slider.multTPCClusterN(0,1.4,0.2,0.2)";
  AliRootInteractive::treeBokehDrawArray("treeSec", query, figureArray, widgets, optionsAll,"drawITSchi2Comparison.html");
}

void drawChi2TPCComparison(){
   TString query = "entries>50";
  TString figureArray= "["
                        "[['dSector'], ['LHC15oChi2TPC'], {'colorZvar':'qPt'}],"
                        "[['dSector'], ['LHC18qChi2TPC'], {'colorZvar':'qPt'}],"
                        "[['dSector'], ['LHC18l8aChi2TPC'], {'colorZvar':'qPt'}],"
                        "['table'] ]";
  TString optionsAll = ""
  "tooltips=[('qPt', '(@qPt)'), ('ps/pt', '(@atgl)'), ('multTPCClusterN', '(@multTPCClusterN)'),"
  "('LHC15oChi2TPC','(@LHC15oChi2TPC)'), ('LHC18qChi2TPC', '(@LHC18qChi2TPC)'),"
  "('LHC18l8aChi2TPC', '(@LHC18l8aChi2TPC)')] ,"
  "layout= '((0, 1,2 ),(3, plot_height=100,x_visible=1),commonX=0, commonY=0,x_visible=1,y_visible=1,plot_height=350,plot_width=1000)'";

  TString widgets="query.xx(),slider.qPt(-3,3,0.3,-0.3,0.3),slider.atgl(0,1,0.1,0.1), slider.dSector(0,1,0.05,0,1),slider.multTPCClusterN(0,1.4,0.2,0.2)";
  AliRootInteractive::treeBokehDrawArray("treeSec", query, figureArray, widgets, optionsAll,"drawITSchi2Comparison.html");
}


