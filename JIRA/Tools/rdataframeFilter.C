/*
  .L ~/github/RootInteractiveTest/JIRA/Tools/rdataframeFilter.C

 */

void makeRDFrameSnapshot0(TTree *tree) {
  auto nCr = [](AliESDtrack &p, Int_t row0, Int_t row1) { return p.GetTPCClusterInfo(3, 1, row0, row1); };
  auto nCl = [](AliESDtrack &p, Int_t row0, Int_t row1) { return p.GetTPCClusterInfo(3, 0, row0, row1); };
  auto nPos = [](AliESDtrack &p, Float_t bz, Int_t row0, Int_t row1) { return p.GetParameterAtRadius(83 + 0.3 * (row0 + row1), bz, 13); };

  ROOT::RDataFrame df(*tree);
  for (Int_t row0 = 0; row0 < 80; row0 += 15) {
    //std::string  z[] ={"esdTrack.",std::to_string(row0), std::to_string(row0+25)};
    // df = df.Define(std::string("cr") + std::to_string(row0), nCr, "esdTrack.", row0, row0 + 1);
  }
}

void makeRDFrameSnapshot0(TTree *tree, const char *outputFile, Float_t bz) { //const char * filter, Long64_t nEvents, Double_t dcaCut) {
  // tree = (TTree *) AliXRDPROOFtoolkit::MakeChain("filtered.list", "highPt", 0, 1000000000, 0, 0);
  // treeEvent = (TTree *) AliXRDPROOFtoolkit::MakeChain("filtered.list", "events", 0, 1000000000, 0, 0);
  //

  auto cr0_25 = [](AliESDtrack &p) { return p.GetTPCClusterInfo(3, 1, 0, 25); };
  auto cr20_45 = [](AliESDtrack &p) { return p.GetTPCClusterInfo(3, 1, 20, 45); };
  auto cr40_65 = [](AliESDtrack &p) { return p.GetTPCClusterInfo(3, 1, 40, 65); };
  auto cr60_85 = [](AliESDtrack &p) { return p.GetTPCClusterInfo(3, 1, 60, 85); };
  //
  auto cf0_25 = [](AliESDtrack &p) { return p.GetTPCClusterInfo(3, 0, 0, 25); };
  auto cf20_45 = [](AliESDtrack &p) { return p.GetTPCClusterInfo(3, 0, 20, 45); };
  auto cf40_65 = [](AliESDtrack &p) { return p.GetTPCClusterInfo(3, 0, 40, 65); };
  auto cf60_85 = [](AliESDtrack &p) { return p.GetTPCClusterInfo(3, 0, 60, 85); };
  //
  auto pos0_25 = [&](AliESDtrack &p) { return p.GetParameterAtRadius(83 + 0.6 * 12, bz, 10); };
  auto pos20_45 = [&](AliESDtrack &p) { return p.GetParameterAtRadius(83 + 0.6 * 32, bz, 10); };
  auto pos40_65 = [&](AliESDtrack &p) { return p.GetParameterAtRadius(83 + 0.6 * 52, bz, 10); };
  auto pos60_85 = [&](AliESDtrack &p) { return p.GetParameterAtRadius(83 + 0.6 * 72, bz, 10); };
  auto  bunchCrossing =[](ULong64_t gid) {return gid&0xFFF;};
  //
  ROOT::RDataFrame *pdf = new ROOT::RDataFrame(*tree);
  auto df = pdf->Filter("abs(esdTrack.fD)<4&&abs(esdTrack.fZ)<3");
  df = df.Define("cr0_25", cr0_25, {"esdTrack."});
  df = df.Define("cr20_45", cr20_45, {"esdTrack."});
  df = df.Define("cr40_65", cr40_65, {"esdTrack."});
  df = df.Define("cr60_85", cr60_85, {"esdTrack."});
  df = df.Define("cf0_25", cf0_25, {"esdTrack."});
  df = df.Define("cf20_45", cf20_45, {"esdTrack."});
  df = df.Define("cf40_65", cf40_65, {"esdTrack."});
  df = df.Define("cf60_85", cf60_85, {"esdTrack."});
  df = df.Define("pos0_25", pos0_25, {"esdTrack."});
  df = df.Define("pos20_45", pos20_45, {"esdTrack."});
  df = df.Define("pos40_65", pos40_65, {"esdTrack."});
  df = df.Define("pos60_85", pos60_85, {"esdTrack."});
  //
  df = df.Define("bunchCrossing",bunchCrossing,{"gid"});
  df = df.Define("qP",[](AliESDtrack &p) { return p.GetSign()/p.P();},{"esdTrack."});
  df = df.Define("qPt",[](AliESDtrack &p) { return p.GetSign()/p.Pt();},{"esdTrack."});
  df = df.Define("P",[](AliESDtrack &p) { return p.P();},{"esdTrack."});
  df = df.Define("Ptpc",[](AliESDtrack &p) { return p.GetInnerParam()->P();},{"esdTrack."});
  df = df.Define("tgl",[](AliESDtrack &p) { return p.GetTgl();},{"esdTrack."});

  //
  std::vector <std::string> columns = {"gid", "bunchCrossing",
                                       "qP","P","Ptpc","tgl","qPt",
                                       "cr0_25","cr20_45", "cr40_65","cr60_85",
                                       "cf0_25","cf20_45", "cf40_65","cf60_85",
                                       "pos0_25","pos20_45", "pos40_65","pos60_85"};
  df.Snapshot("tree", outputFile, columns);
}