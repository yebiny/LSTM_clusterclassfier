from CRABClient.UserUtilities import config

config = config()

config.General.requestName = 'Gen-TT_powheg'
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
# Name of the CMSSW configuration file
config.JobType.psetName = 'genParticles.py'

config.Data.inputDataset = "/TT_TuneCUETP8M2T4_13TeV-powheg-pythia8/RunIISummer16DR80Premix-PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/AODSIM"
config.Data.splitting = 'FileBased'
# config.Data.splitting = 'Automatic'
config.Data.unitsPerJob = 1
config.Data.publication = False
# This string is used to construct the output dataset name
config.Data.outputDatasetTag = 'CRAB3_GEN_TT_powheg'

# Where the output files will be transmitted to
config.Site.storageSite = 'T3_KR_KISTI'
