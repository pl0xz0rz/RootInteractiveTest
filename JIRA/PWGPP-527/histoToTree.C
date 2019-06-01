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