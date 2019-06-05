void anlayzeMaps(){
   TFile::SetCacheFileDir("../data/");
   tree=AliTreePlayer::LoadTrees("cat tree.list",".*_qPt_tgl_phiDist","xxx",".*","","");
   

}