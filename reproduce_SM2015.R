##==============================================================================
## preliminary scratch work for study of optimal journal submission pathways
##
## Questions? Tony Wong (twong@psu.edu)
##==============================================================================

# go to the directory where all the codes and data are
setwd('~/codes/DecisionMaking_JournalSubmission')

# clear the workspace
rm(list=ls())

# install relevant packages
#install.packages('xlsx')
library(xlsx)

# set file name(s) to read/write
filename.data <- '~/codes/DecisionMaking_JournalSubmission/data/Salinas_Munch_2015_S1_Table.xlsx'

# get nice plotting colors: mycol array
source('~/codes/UsefulScripts/R/colorblindPalette.R')

# directory to save the plots
plotdir='~/Box\ Sync/DecisionMaking_JournalSubmission/Figures/'

##==============================================================================
##==============================================================================

# read the data
dat <- read.xlsx(filename.data, 1)






##==============================================================================
##==============================================================================







##==============================================================================
## End
##==============================================================================
