from __future__ import division
from collections import OrderedDict
import numpy as np
from ROOT import TH1F
import os
import matplotlib.pyplot as plt
import numpy
import ROOT, sys

folder = sys.argv[1:][0]

# Load values
info = np.load(folder+"/info_eval.npz")
#info = np.load(folder+"/eval_result.npz")
train_sig_response =info['train_sig_response']
train_bkg_response =info['train_bkg_response']
test_sig_response = info['test_sig_response' ]
test_bkg_response = info['test_bkg_response' ]

# Make canvas
canvas = ROOT.TCanvas("c", "c", 1200, 800)
canvas.cd()
h_test_sig = TH1F("test_sig", "" , 50, 0, 1.)
h_test_bkg = TH1F("test_bkg", "test_bkg" , 50, 0, 1.)
h_train_sig =TH1F("train_sig","train_sig", 50, 0, 1.)
h_train_bkg =TH1F("train_bkg","train_bkg", 50, 0, 1.)
h_test_sig.SetXTitle("Model response")
hists = [h_test_sig, h_test_bkg, h_train_sig, h_train_bkg]
print("save hists")

# Fill
for each in train_sig_response:
    h_train_sig.Fill(each)
print( "fill train sig")
for each in train_bkg_response:
    h_train_bkg.Fill(each)
print( "fill train bkg")
for each in test_sig_response: 
    h_test_sig.Fill(each)
print( "fill test sig"  ) 
for each in test_bkg_response: 
    h_test_bkg.Fill(each)
print("fill test bkg") 

h_test_sig.Scale(1.0/h_test_sig.Integral())
h_test_bkg.Scale(1.0/h_test_bkg.Integral())
h_train_sig.Scale(1.0/h_train_sig.Integral())
h_train_bkg.Scale(1.0/h_train_bkg.Integral())
print("scaleed")

max_value = max(
    h_test_sig.GetMaximum(),
    h_test_bkg.GetMaximum(),
    h_train_sig.GetMaximum(),
    h_train_bkg.GetMaximum(),
)
h_test_sig.SetMaximum(1.4*max_value)

#h_train_sig.SetFillColorAlpha(ROOT.kRed, 0.3)
#h_train_bkg.SetFillColorAlpha(ROOT.kBlue,0.3)
h_train_sig.SetFillStyle(3002)
h_train_bkg.SetFillStyle(3002)
h_train_sig.SetFillColor(ROOT.kOrange+10)
h_train_bkg.SetFillColor(ROOT.kBlue)
h_test_sig.SetLineColor(ROOT.kRed-4)
h_test_bkg.SetLineColor(ROOT.kAzure+7)
h_test_sig.SetLineWidth(4)
h_test_bkg.SetLineWidth(4)

h_test_sig.Draw("HIST")
h_test_bkg.Draw( "HIST same")
h_train_sig.Draw("HIST same")
h_train_bkg.Draw("HIST same")

leg = ROOT.TLegend(0.78,0.75,0.98,.95)
leg.AddEntry(h_test_sig, "Test sig", 'l')
leg.AddEntry(h_test_bkg, "Test bkg", 'l')
leg.AddEntry(h_train_sig,"Train sig",'f')
leg.AddEntry(h_train_bkg,"Train bkg",'f')
leg.Draw("same")

canvas.Draw()
canvas.SaveAs(folder+"/responce.png")
