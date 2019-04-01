#

# Delphes Simulation
Add delphes cards for CMS detector simulation with track smearing.
'delphes_card_CMS_TrackSmearing_AK4.tcl' is a combination of [delphes_card_CMS.tcl](https://github.com/delphes/delphes/blob/master/cards/delphes_card_CMS.tcl) and [trkCountingBTaggingCMS.tcl](https://github.com/delphes/delphes/blob/master/cards/trkCountingBTaggingCMS.tcl)
For 'delphes_card_CMS_TrackSmearing_AK4.tcl',
    Add TrackSmearing module, which is used after 'TrackMerger'.
    Replace 'BTagging' with 'TrackCountingBtagging'.
    Use the jet radius of 0.4
