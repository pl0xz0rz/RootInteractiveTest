TTree * treeMap=0;
TTree *treeTime=0;


void InitMap() {
  TFile::SetCacheFileDir("../data/");
  treeMap = AliTreePlayer::LoadTrees("cat map.list", ".*", ".*Time.*", "", "", "");
  treeTime = AliTreePlayer::LoadTrees("cat map.list", ".*Time.*", "xxx", "", "", "");
}

void egtSomeList(){
  treeTime->GetListOfFriends()->Print("","*tpc*UpCut0*");
}

void fluctuationStudy(){
  treeTime->Draw("LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBPileUpCut0Dist.rmsG:centV0Center","LHC18q.hRelratioSSDtpcTrackBeforeCleanZCentV0TimeTMBPileUpCut0Dist.entries>500");
  treeTime->Draw("LHC18q.hRelratioSSDtpcTrackBeforeCleanZCentV0TimeTMBPileUpCut0Dist.rmsG:centV0Center","LHC18q.hRelratioSSDtpcTrackBeforeCleanZCentV0TimeTMBPileUpCut0Dist.entries>500");
  //
   treeTime->Draw("LHC18q.hRelratioSSDtpcTrackBeforeCleanZCentV0TimeTMBPileUpCut0Dist.rmsG/LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBPileUpCut0Dist.rmsG:centV0Center","LHC18q.hRelratioSSDtpcTrackBeforeCleanZCentV0TimeTMBPileUpCut0Dist.entries>500");


  treeTime->Draw("LHC18q.hRelratioV0tpcMultZCentV0TimeTMBDist.meanG:centV0Center","abs(LHC18q.hRelratioV0tpcMultZCentV0TimeTMBDist.meanG/LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBDist.binMedian-1)<0.2");
  treeTime->Draw("LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBDist.meanG:centV0Center","abs(LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBDist.meanG/LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBDist.binMedian-1)<0.2");


  treeTime->Draw("LHC18q.hRelratioSSDtpcMultZCentV0TimeTMBPileUpCut0Dist.rmsG/LHC18q.hRelratioSSDtpcMultZCentV0TimeTMBPileUpCut0Dist.meanG:centV0Center","abs(LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBDist.meanG/LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBDist.binMedian-1)<0.2");
  treeTime->Draw("LHC18q.hRelratioV0tpcMultZCentV0TimeTMBPileUpCut0Dist.rmsG/LHC18q.hRelratioV0tpcMultZCentV0TimeTMBPileUpCut0Dist.meanG:centV0Center","abs(LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBDist.meanG/LHC18q.hRelratioV0tpcTrackBeforeCleanZCentV0TimeTMBDist.binMedian-1)<0.2");

  AliNDLocalRegression::MakeRegression(treeMap, "rmsGFit", "(5,0,1,5,0.6,1, 5,2,9,5,0,2)", "meanG:0.01", "mdEdxCenter:truncVCenter:logVCenter:mult10000Center", "entries>50&&logVCenter>1", "0.3:0.1:2.1:0.4", 0.01);

}