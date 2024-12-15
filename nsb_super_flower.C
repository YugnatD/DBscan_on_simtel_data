//C, C++
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#include <time.h>

using namespace std;

Int_t nsb_super_flower(){
  //
  TRandom3 *rnd = new TRandom3(13123123);

  Double_t nsb_rate_per_pixel = 0.268; //GHz

  Int_t n_pixels = 49;

  Int_t nEv = 100000;

  Double_t tot_nsb_ADC = 0.0;
  Double_t ADC_per_pe = 8.25;

  TH1D *h1_ADC_tot = new TH1D("h1_ADC_tot","h1_ADC_tot",100,0.0,30);
  
  for(Int_t i = 0;i<nEv;i++){
    tot_nsb_ADC = 0.0;
    for(Int_t j = 0;j<n_pixels;j++){
      if(rnd->Uniform(0.0,1.0)<=nsb_rate_per_pixel)
	tot_nsb_ADC += ADC_per_pe;
    }
    h1_ADC_tot->Fill(tot_nsb_ADC/ADC_per_pe);
  }
  //
  //
  h1_ADC_tot->Draw();
  //
  return 0;
}
