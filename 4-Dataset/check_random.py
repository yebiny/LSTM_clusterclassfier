import ROOT,sys

data_name = sys.argv[1]

root_file = ROOT.TFile(data_name)
tree = root_file.delphys

d0 = tree.AsMatrix(["jet_label"])

print(d0[:100])
print(d0[-100:])
