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

void read_file(TString name, TH1D *h1, TGraphErrors *gr);
void read_root_file(TString name, TH1D *h1, Double_t h1_norm, TH1D *h1_rate,TGraphErrors *gr);

Int_t plots_rates(){
  //
  void read_file(TString name, TH1D *h1, TGraphErrors *gr);
  //
  TH1D *h1_wf_noise = new TH1D();
  h1_wf_noise->SetNameTitle("h1_wf_noise","h1_wf_noise");
  TGraphErrors *gr_wf_noise_vs_thADC = new TGraphErrors();
  gr_wf_noise_vs_thADC->SetNameTitle("gr_wf_noise_vs_thADC","gr_wf_noise_vs_thADC");
  //
  //TH1D *h1_wf_noise_bl = new TH1D();
  //h1_wf_noise_bl->SetNameTitle("h1_wf_noise_bl","h1_wf_noise_bl");
  //TGraphErrors *gr_wf_noise_vs_thADC_bl = new TGraphErrors();
  //gr_wf_noise_vs_thADC->SetNameTitle("gr_wf_noise_vs_thADC_bl","gr_wf_noise_vs_thADC_bl");
  //
  //
  TH1D *h1_isolated_flower;// = new TH1D();
  //h1_isolated_flower->SetNameTitle("h1_isolated_flower","h1_isolated_flower");
  //
  TH1D *h1_isolated_flower_rate;// = new TH1D();
  //h1_isolated_flower_rate->SetNameTitle("h1_isolated_flower_rate","h1_isolated_flower_rate");
  //
  //TGraphErrors *gr_isolated_flower_rate = new TGraphErrors();
  //gr_isolated_flower_rate->SetNameTitle("gr_isolated_flower_rate","gr_isolated_flower_rate");
  //
  //
  //read_file("wf_noise_arr.pdf_rates.pdf.csv", h1_wf_noise, gr_wf_noise_vs_thADC);
  //read_file("wf_noise_blacklist_arr.pdf_rates.pdf.csv", h1_wf_noise_bl, gr_wf_noise_vs_thADC_bl);
  //read_file("wf_noiseNSB268MHz_arr.pdf_rates.pdf.csv", h1_wf_noise, gr_wf_noise_vs_thADC);
  //
  //
  //read_file("L0_digitalsum_noise_arr.pdf_rates.pdf.csv", h1_wf_noise, gr_wf_noise_vs_thADC);
  //read_file("L0_digitalsum_noise_blacklist_arr.pdf_rates.pdf.csv", h1_wf_noise_bl, gr_wf_noise_vs_thADC_bl);
  //read_file("L0_digitalsum_noiseNSB268MHz_arr.pdf_rates.pdf.csv", h1_wf_noise, gr_wf_noise_vs_thADC);

  //
  //read_file("L1_digitalsum_noise_arr.pdf_rates.pdf.csv", h1_wf_noise, gr_wf_noise_vs_thADC);

  //
  //
  //
  //read_file("L3_digitalsum_noiseNSB268MHz_arr_all.pdf_rates.pdf.csv", h1_wf_noise, gr_wf_noise_vs_thADC);
  read_file("L1_digitalsum_noiseNSB268MHz_arr.pdf_rates.pdf.csv", h1_wf_noise, gr_wf_noise_vs_thADC);
  //read_file("L1_max_digi_sum_noiseNSB268MHz.pdf_rates.pdf.csv", h1_wf_noise, gr_wf_noise_vs_thADC);
  //read_file("wf_noiseNSB268MHz_blacklist_arr.pdf_rates.pdf.csv", h1_wf_noise_bl, gr_wf_noise_vs_thADC_bl);
  //read_file("L0_digitalsum_noiseNSB268MHz_blacklist_arr.pdf_rates.pdf.csv", h1_wf_noise_bl, gr_wf_noise_vs_thADC_bl);
  //read_file("L1_digitalsum_noise_blacklist_arr.pdf_rates.pdf.csv", h1_wf_noise_bl, gr_wf_noise_vs_thADC_bl);
  //read_file("L1_digitalsum_noiseNSB268MHz_blacklist_arr.pdf_rates.pdf.csv", h1_wf_noise_bl, gr_wf_noise_vs_thADC_bl);
  //
  TFile *f01 = new TFile("hist_trgB_corsika_run1.root");
  h1_isolated_flower = (TH1D*)f01->Get("h1_digital_sum");
  h1_isolated_flower_rate = (TH1D*)f01->Get("h1_digital_sum_rate");  
  //read_root_file("hist_trgB_corsika_run1.root", h1_isolated_flower, 1.0, h1_isolated_flower_rate, gr_isolated_flower_rate);
  //
  //
  //
  TCanvas *c1 = new TCanvas("c1","c1",10,10,1200,600);
  c1->Divide(2,1);
  gStyle->SetPalette(1);
  gStyle->SetFrameBorderMode(0);
  gROOT->ForceStyle();
  gStyle->SetStatColor(kWhite);
  //gStyle->SetOptStat(kFALSE);
  //  
  //
  //
  //h1_wf_noise_bl->SetLineColor(kBlack);
  //h1_wf_noise_bl->SetLineWidth(3.0);
  //h1_wf_noise->SetLineColor(kRed);
  h1_wf_noise->SetLineColor(kBlack);
  h1_wf_noise->SetLineWidth(3.0);
  //
  //gr_wf_noise_vs_thADC_bl->SetLineColor(kBlack);
  //gr_wf_noise_vs_thADC_bl->SetMarkerColor(kBlack);
  //gr_wf_noise_vs_thADC_bl->SetLineWidth(3.0);
  //
  //gr_wf_noise_vs_thADC->SetLineColor(kRed);
  //gr_wf_noise_vs_thADC->SetMarkerColor(kRed);
  gr_wf_noise_vs_thADC->SetLineColor(kBlack);
  gr_wf_noise_vs_thADC->SetMarkerColor(kBlack);
  gr_wf_noise_vs_thADC->SetLineWidth(3.0);
  //
  //gr_wf_noise_vs_thADC->Draw("APL");
  //
  //h1_weight_01->Fit(f_power_law, "",  "", Emin, Emax);
  //
  //h1_weight_01->Draw();
  //
  //
  c1->cd(1);
  gPad->SetLogy();
  gPad->SetGridx();
  gPad->SetGridy();
  //h1_wf_noise_bl->SetTitle("");
  //h1_wf_noise_bl->Draw();
  //h1_wf_noise->Draw("sames");
  h1_wf_noise->SetTitle("");
  h1_wf_noise->Draw();
  //h1_isolated_flower->Draw("sames");
  //
  //h1_wf_noise_bl->GetXaxis()->SetTitle("ADC counts");
  h1_wf_noise->GetXaxis()->SetTitle("ADC counts");
  //
  c1->cd(2);
  //
  gPad->SetLogy();
  gPad->SetGridx();
  gPad->SetGridy();
  //
  TMultiGraph *mg = new TMultiGraph();
  mg->Add(gr_wf_noise_vs_thADC);
  //mg->Add(gr_wf_noise_vs_thADC_bl);
  //
  //mg->Add(gr_wf_noise_vs_thADC);
  //mg->Add(gr_wf_noise_vs_thADC_bl);
  //
  mg->Draw("apl");
  //h1_isolated_flower_rate->Draw("sames");
  //
  mg->SetMaximum(1.0e+14);
  mg->SetMinimum(1.0e+1);
  //
  mg->GetXaxis()->SetTitle("ADC counts");
  mg->GetYaxis()->SetTitle("Rate, Hz");

  //  
  /*
  TLegend *leg = new TLegend(0.6,0.6,0.9,0.9,"","brNDC");
  leg->AddEntry(gr, "simtelarr", "apl");
  leg->AddEntry(gr_sim, "sim", "apl");
  leg->Draw();  
  */
  /*
  h1_Ew->SetLineColor(kRed);
  h1_Ew->SetLineWidth(3.0);  
  h1_E->SetLineColor(kBlack);
  h1_E->SetLineWidth(3.0);  
  //
  h1_E->Draw();
  h1_Ew->Draw("same");
  //
  TCanvas *c2 = new TCanvas("c2","c2",10,10,600,600);
  gStyle->SetPalette(1);
  gStyle->SetFrameBorderMode(0);
  gROOT->ForceStyle();
  gStyle->SetStatColor(kWhite);
  //gStyle->SetOptStat(kFALSE);
  gr_w_vs_E->Fit(f_weight, "",  "", Emin, Emax);
  gr_w_vs_E->Draw("AP");  
  //
  */
  return 0;
}

void read_file(TString name, TH1D *h1, TGraphErrors *gr){
  std::ifstream fFile(name.Data());
  TString mot;
  Double_t thresholds, counts, rates;
  Double_t thresholdsf, countsf, ratesf;
  Double_t binwidth;
  Double_t binx, biny;
  //
  Int_t counter = 0;
  //
  TGraph *gr_tmp = new TGraph();
  //
  //thresholds counts rates;
  //259.0 0.0 8178684488340.191
  //wf_noise_arr.pdf_rates.pdf.csv
  //
  if(fFile.is_open()){
    fFile>>mot>>mot>>mot;
    fFile>>thresholdsf>>countsf>>ratesf;
    counter++;
    while(fFile>>thresholds>>counts>>rates){
      if(counter == 1){
	binwidth = thresholds - thresholdsf;
	gr->SetPoint(gr->GetN(),thresholdsf + binwidth/2,ratesf);
	gr->SetPointError(gr->GetN()-1,0.0,rates*0.0);
	gr_tmp->SetPoint(gr_tmp->GetN(),thresholdsf + binwidth/2,countsf);
      }
      gr->SetPoint(gr->GetN(),thresholds + binwidth/2,rates);
      gr->SetPointError(gr->GetN()-1,0.0,rates*0.0);
      gr_tmp->SetPoint(gr_tmp->GetN(),thresholds + binwidth/2,counts);
      counter++;
    }
    fFile.close();
  }
  //
  Double_t binmin, binmax;
  Int_t nbins = counter;
  //
  gr_tmp->GetPoint( 0, binx, biny);
  binmin = binx - binwidth/2-0;
  gr_tmp->GetPoint((gr_tmp->GetN()-1), binx, biny);
  binmax = binx + binwidth/2-0;
  //
  h1->SetBins( nbins, binmin, binmax);
  for(Int_t i = 0; i<gr_tmp->GetN(); i++){
    gr_tmp->GetPoint(i, binx, biny);
    h1->SetBinContent(i+1,biny);
  }
}

void read_root_file(TString name, TH1D *h1, Double_t h1_norm, TH1D *h1_rate, TGraphErrors *gr){
  //
  TFile *f01 = new TFile(name.Data());
  //
  h1 = (TH1D*)f01->Get("h1_digital_sum");
  h1_rate = (TH1D*)f01->Get("h1_digital_sum_rate");  
}
