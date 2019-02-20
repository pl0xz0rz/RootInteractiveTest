/*

 */
#include "TTreeStream.h"
#include "TRandom.h"
#include "TMath.h"
#include "TDatabasePDG.h"
#include "TROOT.h"
#include "TF1.h"
#include "AliTMinuitToolkit.h"
#include "TMatrixD.h"

Double_t mass[4]={};
TTree* tree =0 ;
Double_t paramBB[5]={0.76176, 10.632, 1.3279, 1.8631, 1.9479};
TVectorD vectorBB(5,paramBB);

AliTMinuitToolkit *fitterBBABG=0;

///his is the empirical ALEPH parameterization of the Bethe-Bloch formula.
//  // It is normalized to 1 at the minimum. From the  AliExternalTrackParam
//  // bg - beta*gamma
//  // The default values for the kp* parameters are for ALICE TPC.
//  // The returned value is in MIP units
/// \return
Double_t BetheBlochAleph(Double_t bg, Double_t kp1, Double_t kp2, Double_t kp3, Double_t kp4, Double_t kp5){//
  Double_t beta = bg/TMath::Sqrt(1.+ bg*bg);
  Double_t aa = TMath::Power(beta,kp4);
  Double_t bb = TMath::Power(1./bg,kp5);
  bb=TMath::Log(TMath::Abs(kp3*0.00001+bb));
  return (kp2-aa-bb)*kp1*0.1/aa;
}


void simulateBBTree(Int_t nPoints=100000){
  TTreeSRedirector *pcstream = new TTreeSRedirector("dEdx.root");
  Double_t pMin=0.2, pMax=20;
  mass[0]=TDatabasePDG::Instance()->GetParticle("e-")->Mass();
  mass[1]=TDatabasePDG::Instance()->GetParticle("pi-")->Mass();
  mass[2]=TDatabasePDG::Instance()->GetParticle("K-")->Mass();
  mass[3]=TDatabasePDG::Instance()->GetParticle("proton")->Mass();
  for (Int_t iPoint=0; iPoint<nPoints; iPoint++){
    Double_t momentum=0;
    if (iPoint%2==0) momentum=pMin+gRandom->Rndm()*pMax;
    if (iPoint%2==1) momentum=pMin/(gRandom->Rndm()+0.001);
    Int_t type = gRandom->Rndm()*4;
    Double_t betaGamma=momentum/mass[type];
    Double_t bb=BetheBlochAleph(betaGamma, paramBB[0], paramBB[1], paramBB[2], paramBB[3], paramBB[4] );
    Double_t noise= bb*gRandom->Gaus()*0.05;
    (*pcstream)<<"dEdx"<<
      "p="<<momentum<<
      "type="<<type<<
      "bg="<<betaGamma<<
      "bb="<<bb<<
      "noise="<<noise<<
      "\n";
  }
  delete pcstream;
}


void InitTree(){
  AliTMinuitToolkit::RegisterPlaneFitter(2,2);
  TFile *f =TFile::Open("dEdx.root");
  tree = (TTree*)f->Get("dEdx");
  TFormula *formulaBBABG = new TFormula("BBABG", "[0]*BetheBlochAleph(x[0],[1],[2],[3],[4],[5])");
  TMatrixD  * initParamBBABG= new TMatrixD(6,4), &paramBBABG=*initParamBBABG;
  fitterBBABG= new AliTMinuitToolkit("fitterBBABG.root");
  fitterBBABG->SetFitFunction((TF1 *) formulaBBABG, kTRUE);
  fitterBBABG->SetVerbose(0x1);
  fitterBBABG->SetLogLikelihoodFunction((TF1*)(gROOT->FindObject("likePseudoHuber")));
  paramBBABG(0,0)=1;  paramBBABG(0,1)=0.1;
  for (Int_t i=1; i<6; i++) {
    paramBBABG(i,0)=paramBB[i-1];
    paramBBABG(i,1)=paramBB[i-1]*0.1;
  }
  fitterBBABG->SetInitialParam(&paramBBABG);
}


void FitBBABG(){
   fitterBBABG->FillFitter(tree, "bb+noise:100","bg","bg>0.01",0,10000);
  fitterBBABG->Bootstrap(10,"bbabg");
  tree->SetAlias("bbfit",fitterBBABG->GetFitFunctionAsAlias());

}