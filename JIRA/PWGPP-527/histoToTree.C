/// convert histograms to tree
void histoToTree(){
  TFile *fMC=TFile::Open("MC_Phi_LHC16qt.root");
  TFile *fData=TFile::Open("Data_Phi_LHC16qt.root");
  TTreeSRedirector *pcstream=new TTreeSRedirector("conversionTree.root","recreate");

  for (Int_t iKey=0; iKey<fMC->GetListOfKeys()->GetEntries(); iKey++){
    TString name="MC_";
    name+=fMC->GetListOfKeys()->At(iKey)->GetName();
    name+=".=";
    TObject *o=fMC->Get(fMC->GetListOfKeys()->At(iKey)->GetName());
    printf("%s\n",name.Data());
    (*pcstream)<<"conv"<<name.Data()<<o;
  }
 for (Int_t iKey=0; iKey<fData->GetListOfKeys()->GetEntries(); iKey++){
    TString name="Data_";
    name+=fData->GetListOfKeys()->At(iKey)->GetName();
    name+=".=";
    TObject *o=fData->Get(fData->GetListOfKeys()->At(iKey)->GetName());
    printf("%s\n",name.Data());
    (*pcstream)<<"conv"<<name.Data()<<o;
  }
  (*pcstream)<<"conv"<<"\n";
  delete pcstream;

}

void mergeBigHn(){
  TObjArray* flist = gSystem->GetFromPipe("cat mc.list").Tokenize("\n");
  THn *his = 0;
  //for (Int_t i=0; i<flist->GetEntries(); i++){
  for (Int_t i=0; i<2; i++){
    printf("%s\n",flist->At(i)->GetName());
    TFile * f=TFile::Open(flist->At(i)->GetName());
    THnSparseF* hisAdd0 = (THnSparseF*)f->Get("hR_phi_Z_pT");
    // first project pt 0-2 GeV
    Int_t projection[4]={0,1,2,3};
    Int_t rebin2[4]={2,2,10,10};
    hisAdd0->GetAxis(3)->SetRangeUser(0,2);
    THnSparseF* hisAdd1=(THnSparseF*)hisAdd0->ProjectionND(4,projection);
    delete hisAdd0;
    THnSparseF* hisAdd2=(THnSparseF*)hisAdd1->Rebin(rebin2);
    delete hisAdd1;
    if (his==NULL){
      his=THn::CreateHn("merge","merge", hisAdd2);
    }else{
      THn* hisTMP=THn::CreateHn("merge","merge", hisAdd2);
      his->Add(hisTMP);
      delete hisTMP;
    }
    delete hisAdd2;
  }
  TFile *fout = TFile::Open("mcMerged.root","recreate");
  his->Write("hR_phi_Z_pT");
  fout->Close();
}