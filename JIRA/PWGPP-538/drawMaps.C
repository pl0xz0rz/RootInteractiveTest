/*

  .L $AliPhysics_SRC/PWGPP/AliRootInteractive.h
  .L drawMaps.C


*/

TTree * treeMap=0;

TString html="";

void LoadFits(const char *path, TString prefix){
  TFile::SetCacheFileDir("../data/");
  TFile *fin=TFile::Open(path,"cacheread");
  TList * flist= fin->GetListOfKeys();
  TString fitName;
  AliNDLocalRegression *regression=0;
  for (Int_t i=0; i<flist->GetEntries(); i++){
    ::Info("LoadFits","%s", flist->At(i)->GetName());
    fitName=flist->At(i)->GetName();
    regression  = (AliNDLocalRegression *)fin->Get(fitName.Data());
    Int_t hashIndex=regression->GetVisualCorrectionIndex(regression->GetName());
    AliNDLocalRegression::AddVisualCorrection(regression, hashIndex);
    treeMap->SetAlias(prefix+regression->GetName(),TString::Format("AliNDLocalRegression::GetCorrND(%d,%s.shiftMCenter,%s.nPileUpPrimCenter,%s.primMultCenter,%s.atglCenter+0)",
            hashIndex, prefix.Data(),prefix.Data(),prefix.Data(),prefix.Data()).Data());
  }
}



void initTree(){
  //f=TFile::Open("https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18q/pass1New/mapdEdx.root","cacheread");
  // f=TFile::Open("https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18r/pass1New/mapdEdx.root","cacheread");
  TFile::SetCacheFileDir("../data/");
  treeMap = AliTreePlayer::LoadTrees("cat mapNew.list", ".*", ".*_mdEdxDist*", "", "", "");
  //
  treeMap->SetAlias("shift","shiftMCenter");
  treeMap->SetAlias("nPileUp","nPileUpPrimCenter");
  treeMap->SetAlias("tgl","atglCenter");
  treeMap->SetAlias("nPrim","primMultCenter");
  //
  treeMap->SetAlias("LHC15oOK","(LHC15o_pass1.hdEdxAShifttMNTglDist.binMedian!=0)*3");
  treeMap->SetAlias("LHC18qOK","(LHC18q_pass1.hdEdxAShifttMNTglDist.binMedian!=0)*3");
  treeMap->SetAlias("LHC18rOK","(LHC18r_pass1.hdEdxAShifttMNTglDist.binMedian!=0)*3");
  //
  treeMap->SetAlias("LHC15oMG","LHC15o_pass1.hdEdxAShifttMNTglDist.meanG");
  treeMap->SetAlias("LHC18qMG","LHC18q_pass1.hdEdxAShifttMNTglDist.meanG");
  treeMap->SetAlias("LHC18rMG","LHC18r_pass1.hdEdxAShifttMNTglDist.meanG");
  //
  treeMap->SetAlias("LHC15oMed","LHC15o_pass1.hdEdxAShifttMNTglDist.binMedian");
  treeMap->SetAlias("LHC18qMed","LHC18q_pass1.hdEdxAShifttMNTglDist.binMedian");
  treeMap->SetAlias("LHC18rMed","LHC18r_pass1.hdEdxAShifttMNTglDist.binMedian");
  //
  treeMap->SetAlias("LHC15oMedA","LHC15o_pass1.hdEdxAShifttMNTglDist.binMedian");
  treeMap->SetAlias("LHC18qMedA","LHC18q_pass1.hdEdxAShifttMNTglDist.binMedian");
  treeMap->SetAlias("LHC18rMedA","LHC18r_pass1.hdEdxAShifttMNTglDist.binMedian");
  //
  treeMap->SetAlias("LHC15oMedC","LHC15o_pass1.hdEdxCShifttMNTglDist.binMedian");
  treeMap->SetAlias("LHC18qMedC","LHC18q_pass1.hdEdxCShifttMNTglDist.binMedian");
  treeMap->SetAlias("LHC18rMedC","LHC18r_pass1.hdEdxCShifttMNTglDist.binMedian");
  LoadFits("https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18q/pass1New/dEdxFitLight.root","LHC15o_pass1.");
  LoadFits("https://rootinteractive.web.cern.ch/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18q/pass1New/dEdxFitLight.root","LHC15o_pass1.");
}




void makeDrawMG(){
  if (treeMap==0) initTree();
  TString query = "entries>0";
  TString figureArray= "["
                        "[['shift'], ['LHC15oMG'], {\"size\": 5, 'colorZvar':'nPileUp'}],"
                        "[['shift'], ['LHC18qMG'], {\"size\": 5, 'colorZvar':'nPileUp'}],"
                        "[['shift'], ['LHC18rMG'], {\"size\": 5, 'colorZvar':'nPileUp'}],"
                        "['table'] ]";
  TString optionsAll = "tooltips=[('shift', '(@shift)'), ('nPrim', '(@nPrim)'), ('nPileUp', '(@nPileUp)'),('tgl', '(@tgl)')] ,"
                    "layout= '((0, 1,2 ),(3, plot_height=100,x_visible=1),commonX=0, x_visible=1,y_visible=1,plot_height=350,plot_width=1500)'";
  TString widgets="query.xx(),slider.shift(-180,180,20,-180,180),slider.nPrim(0,3000,300,300), slider.tgl(0,1,0.1,0.1)";
  AliRootInteractive::treeBokehDrawArray("treeMap", query, figureArray, widgets, optionsAll,0);
}


void makeDrawMed(){
  if (treeMap==0) initTree();
  TString query = "entries>0";
  TString figureArray= "["
                        "[['shift'], ['LHC15oMed'], {\"size\": 'LHC15oOK', 'colorZvar':'nPileUp'}],"
                        "[['shift'], ['LHC18qMed'], {\"size\": 'LHC18qOK', 'colorZvar':'nPileUp'}],"
                        "[['shift'], ['LHC18rMed'], {\"size\": 'LHC18rOK', 'colorZvar':'nPileUp'}],"
                        "['table'] ]";
  TString optionsAll = "tooltips=[('shift', '(@shift)'), ('nPrim', '(@nPrim)'), ('nPileUp', '(@nPileUp)'),('tgl', '(@tgl)'), ('status', '(@LHC15oOK,@LHC18qOK,@LHC18rOK)')] ,"
                    "layout= '((0, 1,2 ),(3, plot_height=100,x_visible=1),commonX=0, x_visible=1,y_visible=1,plot_height=350,plot_width=1500)'";
  TString widgets="query.xx(),slider.shift(-180,180,20,-180,180),slider.nPileUp(0,7000,500,0,7000),slider.nPrim(0,3000,300,300), slider.tgl(0,1,0.1,0.1)";
  AliRootInteractive::treeBokehDrawArray("treeMap", query, figureArray, widgets, optionsAll,0);
}

void makeDrawMedMap(){
  if (treeMap==0) initTree();
  TString query = "entries>0";
  TString figureArray= "["
                        "[['shift'], ['LHC15oMed'], {\"size\": 'LHC15oOK', 'colorZvar':'nPileUp'}],"
                        "[['shift'], ['LHC18qMed'], {\"size\": 'LHC18qOK', 'colorZvar':'nPileUp'}],"
                        "[['shift'], ['LHC18rMed'], {\"size\": 'LHC18rOK', 'colorZvar':'nPileUp'}],"
                        "[['LHC15oMedA'], ['LHC15oMedC'], {\"size\": 'LHC15oOK', 'colorZvar':'nPileUp'}],"
                        "[['LHC18qMedA'], ['LHC18qMedC'], {\"size\": 'LHC18qOK', 'colorZvar':'nPileUp'}],"
                        "[['LHC18rMedA'], ['LHC18rMedC'], {\"size\": 'LHC18rOK', 'colorZvar':'nPileUp'}],"
                        "['table'] ]";
  TString optionsAll = "tooltips=[('shift', '(@shift)'), ('nPrim', '(@nPrim)'), ('nPileUp', '(@nPileUp)'),('tgl', '(@tgl)'), ('status', '(@LHC15oOK,@LHC18qOK,@LHC18rOK)')] ,"
                    "layout= '((0, 1,2, commonX=0),(3, 4 ,5,  commonX=3,plot_height=250), (6, plot_height=100,x_visible=1), x_visible=1,y_visible=1,plot_height=300,plot_width=1500)'";
  TString widgets="query.xx(),slider.shift(-180,180,20,-180,180),slider.nPileUp(0,7000,500,0,7000),slider.nPrim(0,3000,300,300), slider.tgl(0,1,0.1,0.1)";
  AliRootInteractive::treeBokehDrawArray("treeMap", query, figureArray, widgets, optionsAll,"drawMaps_makeDrawMedMap.html");
}










//
//void copyData(){
//  rsync -avzt --progeess  mivanov@lxplus.cern.ch:/eos/user/r/rootinteractive/www/testData/JIRA/PWGPP-538/alice/data/2018/LHC18q/pass1New/mapdEdx.root \
//  /home2/miranov/github/RootInteractiveTest/JIRA/data/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18q/pass1New/
//  rsync -avzt mivanov@lxplus.cern.ch:/eos/user/r/rootinteractive/www/testData/JIRA/PWGPP-538/alice/data/2018/LHC18r/pass1New/mapdEdx.root \
//  /home2/miranov/github/RootInteractiveTest/JIRA/data/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18r/pass1New/
//   rsync -avzt --progress  mivanov@lxplus.cern.ch:/eos/user/r/rootinteractive/www/testData/JIRA/PWGPP-538/alice/data/2015/LHC15o/pass1/mapdEdx.root \
//  /home2/miranov/github/RootInteractiveTest/JIRA/data/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2015/LHC15o/pass1/
//  rsync -avzt --progress  mivanov@lxplus.cern.ch:/eos/user/r/rootinteractive/www/testData/JIRA/PWGPP-538/alice/data/2018/LHC18q/pass1New/dEdxFitLight.root \
//  /home2/miranov/github/RootInteractiveTest/JIRA/data/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18q/pass1New/
//  rsync -avzt --progress  mivanov@lxplus.cern.ch:/eos/user/r/rootinteractive/www/testData/JIRA/PWGPP-538/alice/data/2018/LHC18r/pass1New/dEdxFitLight.root \
//  /home2/miranov/github/RootInteractiveTest/JIRA/data/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2018/LHC18r/pass1New/
//  rsync -avzt --progress  mivanov@lxplus.cern.ch:/eos/user/r/rootinteractive/www/testData/JIRA/PWGPP-538/alice/data/2015/LHC15o/pass1/dEdxFitLight.root \
//  /home2/miranov/github/RootInteractiveTest/JIRA/data/RootInteractive/testData/JIRA/PWGPP-538/alice/data/2015/LHC15o/pass1/
//}