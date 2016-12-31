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
N.jour <- nrow(dat)   # number of journals total (can restrict to specialized set)

# set parameters
T <- 5*365  # time horizon [days]
tR <- 30    # time in review [days]
s <- 0.001  # scooping probability

##==============================================================================
##==============================================================================

# define a function to compare starting with journal j over journal k
# (their equation 2) If score(j,k) >= score(k,j), then j is a better choice.
# NB: all times should be in days; impact factors are scaled by 1/365 to be
# expected citations per day.
score <- function(accept_j,time_j,if_j,accept_k,time_k,if_k,T,tR,s){
  out <- accept_j*(if_j/365)*( T-time_j-(1-accept_k)*((1-s)^(tR+time_k))*accept_k*(if_k/365)*(T-time_k) +
                              (1-accept_k)*((1-s)^(tR+time_k))*
                              accept_j*(if_j/365)*(T-time_k-tR-time_j) )
  return(out)
}

score.supp <- function(accept_j,time_j,if_j,accept_k,time_k,if_k,T,tR,s){
  out <- accept_j*(if_j/365)*(T-time_j) +
         (1-accept_j)*((1-s)^(tR+time_j))*accept_k*(if_k/365)*(T-time_j-tR-time_k)
  return(out)
}

# Define function for total expected citations
# Requires a submission pathway (jour_seq) and the acceptance rates (accept_seq),
# time to decision (time_seq), and impact factor (if_seq) for each journal along
# that pathway. qnew at j is basically the snippet of probability that your article
# ends up in journal j, and all the qnew need to sum to 1 to get a true probability
# distribution. This applies to "submissions" and "taccept" below, as well.
# NB: in the original work (SM2015), eq. 1 contains sum(tau_j,k=1:j). This ought
# to be sum(tau_k,k=1:j), no?
citations <- function(jour_seq, accept_seq, time_seq, if_seq, T, tR, s){
  tot <- 0
  qtot <- 0   # normalization constant, to make this a real probability distribution
  j <- 1      # separate, to address the product
  qnew <- accept_seq[j]
  new <- (if_seq[j]/365) * max(0, T-time_seq[j]) * qnew
  tot <- tot+new
  qtot <- qtot+qnew
  if(length(jour_seq)>1){
    for (j in 2:length(jour_seq)){
      qnew <- accept_seq[j] * prod( (1-accept_seq[1:(j-1)])*((1-s)^(time_seq[1:(j-1)]+tR)) )
      new <- (if_seq[j]/365) * max(0, T-sum(time_seq[1:j])-(j-1)*tR) * qnew
      tot <- tot+new
      qtot <- qtot+qnew
    }
  }
  #return(tot)
  return(tot/qtot)
}

# Define function for total expected submissions
# Requires a submission pathway (jour_seq) and the acceptance rates (accept_seq)
# and time to decision (time_seq) for each journal along that pathway. This does
# not include their normalization constant q (eq. 4)
submissions <- function(jour_seq, accept_seq, time_seq, T, tR, s){
  tot <- 0
  qtot <- 0   # normalization constant, to make this a real probability distribution
  j <- 1      # separate, to address the product
  qnew <- accept_seq[j]
  new <- j * 0.5*(sign(T-time_seq[1]) + 1) * qnew
  tot <- tot+new
  qtot <- qtot+qnew
  if(length(jour_seq)>1){
    for (j in 2:length(jour_seq)){
      qnew <- accept_seq[j] * prod( (1-accept_seq[1:(j-1)])*((1-s)^(time_seq[1:(j-1)]+tR)) )
      new <- j * 0.5*(sign( T - sum(time_seq[1:j]) - (j-1)*tR) + 1) * qnew
      tot <- tot+new
      qtot <- qtot+qnew
    }
  }
  #return(tot)
  return(tot/qtot)
}

# Define function for total expected time to acceptance
# Requires a submission pathway (jour_seq) and the acceptance rates (accept_seq),
# time to decision (time_seq), and impact factor (if_seq) for each journal along
# that pathway. This does not include their normalization constant q (eq. 5)
taccept <- function(jour_seq, accept_seq, time_seq, T, tR, s){
  tot <- 0
  qtot <- 0   # normalization constant, to make this a real probability distribution
  j <- 1      # separate, to address the product
  qnew <- accept_seq[j]
  new <- time_seq[j] * 0.5*(sign(T-time_seq[j])+1) * qnew
  tot <- tot+new
  qtot <- qtot+qnew
  if(length(jour_seq)>1){
    for (j in 2:length(jour_seq)){
      qnew <- accept_seq[j] * prod( (1-accept_seq[1:(j-1)])*((1-s)^(time_seq[1:(j-1)]+tR)) )
      new <- (time_seq[j]+(j-1)*tR) * 0.5*(sign(T-sum(time_seq[1:j])-(j-1)*tR)+1) * qnew
      tot <- tot+new
      qtot <- qtot+qnew
    }
  }
  #return(tot)
  return(tot/qtot)
}

# calculate their V_j (equation 3)
# Does not matter that IF is in citations/year and times are all in days because
# the times are considered as ratios, and s is *daily* probability of being scooped.
V <- dat[,'AcceptRate']*dat[,'IF_2012']*(1-dat[,'SubToDecTime_days']/T) /
     ( 1 - (1 - dat[,'SubToDecTime_days']/T - tR/T)*(1-dat[,'AcceptRate'])*((1-s)^(tR+dat[,'SubToDecTime_days'])) )

##==============================================================================
##==============================================================================

# Model 1: Maximize citations only (eq. 2, or alternatively, SOM equation)

# Apply "score" to all pairs of journals and the optimal journal is the one that
# dominates the greatest number of other journals (j dominates k if score(j,k) > score(k,j)).
# After each selection (of submitting to journal j*), T is reduced by time[j*]+tR,
# and the "score" comparison begins again with journal j* removed from the list.

# set parameters for matching SM2015 Fig 3.
T <- 30*365
s <- 0.001
tR <- 30

# initialize
dat_pool <- dat               # initialize dat for only the pool of candidate journals
pool <- 1:nrow(dat_pool)      # initialize indices of pool of possible journals
T_pool <- T                   # initialize total time left on time horizon
isub <- rep(NA,length(pool))  # initialize the indices of the submission sequence

# iterate over total number of journals
for (n in 1:nrow(dat)) {
  ndom <- rep(0,length(pool))   # initialize number of other journals each dominates

  for (j in 1:length(pool)){
    for (k in 1:length(pool)){
      score_jk <- score(dat_pool[j,'AcceptRate'],dat_pool[j,'SubToDecTime_days'],dat_pool[j,'IF_2012'],
                        dat_pool[k,'AcceptRate'],dat_pool[k,'SubToDecTime_days'],dat_pool[k,'IF_2012'],T_pool,tR,s)
      score_kj <- score(dat_pool[k,'AcceptRate'],dat_pool[k,'SubToDecTime_days'],dat_pool[k,'IF_2012'],
                        dat_pool[j,'AcceptRate'],dat_pool[j,'SubToDecTime_days'],dat_pool[j,'IF_2012'],T_pool,tR,s)
#      score_jk <- score.supp(dat_pool[j,'AcceptRate'],dat_pool[j,'SubToDecTime_days'],dat_pool[j,'IF_2012'],
#                             dat_pool[k,'AcceptRate'],dat_pool[k,'SubToDecTime_days'],dat_pool[k,'IF_2012'],T_pool,tR,s)
#      score_kj <- score.supp(dat_pool[k,'AcceptRate'],dat_pool[k,'SubToDecTime_days'],dat_pool[k,'IF_2012'],
#                             dat_pool[j,'AcceptRate'],dat_pool[j,'SubToDecTime_days'],dat_pool[j,'IF_2012'],T_pool,tR,s)
      if(score_jk > score_kj){ndom[j] <- ndom[j]+1}  # okay not to check j==k, because strict inequality rules this out
    }
  }

  # remove the dominant journal (highest ndom), reduce pool, dat_pool, T_pool
  idom <- which(ndom==max(ndom))
  if(length(idom)==1){
    # case where a single journal dominates
    isub[n] <- which(dat[,1]==dat_pool[idom,'Journal']) # find the journal index out of the original pool
    pool <- pool[-idom]
    T_pool <- T_pool - tR - dat_pool[idom,'SubToDecTime_days']
    dat_pool <- dat_pool[-idom,]
  } else {
    # case where two journals are tied
    for (i in 1:length(idom)){
      isub[n+i-1] <- which(dat[,1]==dat_pool[idom[i],'Journal'])
      T_pool <- T_pool - tR - dat_pool[idom[i],'SubToDecTime_days']
    }
    pool <- pool[-idom]
    dat_pool <- dat_pool[-idom,]
  }

  # check that the time horizon has not expired
  if(T_pool <= 0){break}
}

##==============================================================================
##==============================================================================

# Model 2: balancing citations and frustrations

# set parameters for matching SM2015 Fig 3.
T <- 5*365
s <- 0.002
tR <- 30

nchain <- N*(N-1)     # ensure each combination of first two journals is covered
niter <- 20          # number of proposals for each chain
indices <- 1:N
jseq <- array(NA,c(nchain,niter,N))   # to hold journal sequences
cite <- mat.or.vec(nchain,niter)      # to hold the expected citation counts for each pathway
subs <- mat.or.vec(nchain,niter)      # to hold the expected number of submissions for each pathway
tacc <- mat.or.vec(nchain,niter)      # to hold the expected time in review for each pathway

# sample indices for first journal
first <- as.vector(t(matrix(rep(indices,ceiling(nchain/N)),ncol=ceiling(nchain/N))))
#first <- rep(46,length(first))

# sample indices for second journal
second <- rep(NA,nchain)
for (i in 1:N) {second[((i-1)*(N-1)+1):(i*(N-1))] <- indices[-i]}

# fill in first two journals. only need to fill in the first iteration (3rd element)
jseq[,1,1] <- first
jseq[,1,2] <- second

# sample the rest. again, only need the first iteration (3rd element)
for (n in 1:nchain){
  jseq[n,1,3:N] <- sample(indices[-jseq[n,1,1:2]],size=N-2,replace=FALSE)
}
# Test sampling second through end journals
#for (n in 1:nchain){
#  jseq[n,1,2:N] <- sample(indices[-jseq[n,1,1]],size=N-1,replace=FALSE)
#}

# for each chain, propose niter sequence swaps. accept with probability
# alpha = (C.prop/C.curr)*(R.curr/R.prop)
# This set-up implies that if the proposal is better (more citations and/or fewer
# submissions) than the current journal, then it is accepted with probability 1.

t0 <- proc.time()
pb <- txtProgressBar(min=0,max=nchain,initial=0,style=3)
for (n in 1:nchain){
  cite.curr <- citations(jour_seq=jseq[n,1,],
                         accept_seq=dat[jseq[n,1,],'AcceptRate'],
                         time_seq=dat[jseq[n,1,],'SubToDecTime_days'],
                         if_seq = dat[jseq[n,1,],'IF_2012'],
                         T=T, tR=tR, s=s)
  subs.curr <- submissions(jour_seq=jseq[n,1,],
                           accept_seq=dat[jseq[n,1,],'AcceptRate'],
                           time_seq=dat[jseq[n,1,],'SubToDecTime_days'],
                           T=T, tR=tR, s=s)
  tacc.curr <- taccept(jour_seq=jseq[n,1,],
                       accept_seq=dat[jseq[n,1,],'AcceptRate'],
                       time_seq=dat[jseq[n,1,],'SubToDecTime_days'],
                       T=T, tR=tR, s=s)
  cite[n,1] <- cite.curr
  subs[n,1] <- subs.curr
  tacc[n,1] <- tacc.curr
  for (i in 2:niter){
    iswap <- sample(indices,size=2,replace=FALSE)
    jseq.prop <- jseq[n,i-1,]; jseq.prop[iswap[1]] <- jseq[n,i-1,iswap[2]]; jseq.prop[iswap[2]] <- jseq[n,i-1,iswap[1]];
    cite.prop <- citations(jour_seq=jseq.prop,
                           accept_seq=dat[jseq.prop,'AcceptRate'],
                           time_seq=dat[jseq.prop,'SubToDecTime_days'],
                           if_seq = dat[jseq.prop,'IF_2012'],
                           T=T, tR=tR, s=s)
    subs.prop <- submissions(jour_seq=jseq.prop,
                             accept_seq=dat[jseq.prop,'AcceptRate'],
                             time_seq=dat[jseq.prop,'SubToDecTime_days'],
                             T=T, tR=tR, s=s)
    tacc.prop <- taccept(jour_seq=jseq.prop,
                         accept_seq=dat[jseq.prop,'AcceptRate'],
                         time_seq=dat[jseq.prop,'SubToDecTime_days'],
                         T=T, tR=tR, s=s)
    p.accept <- min(1, (cite.prop/cite.curr)*(subm.curr/subm.prop)*(tacc.curr/tacc.prop) )
#    p.accept <- min(1, (cite.prop/cite.curr)*(subm.curr/subm.prop) )
#    p.accept <- min(1, (cite.prop/cite.curr) )
    draw <- runif(1)
    if(draw < p.accept) {
      # accept proposal
      cite.curr <- cite.prop
      subs.curr <- subs.prop
      tacc.curr <- tacc.prop
      jseq[n,i,] <- jseq.prop
    } else {
      # reject proposal, stick with current iterate
      jseq[n,i,] <- jseq[n,i-1,]
    }
    cite[n,i] <- cite.curr
    subs[n,i] <- subs.curr
    tacc[n,i] <- tacc.curr
  }
  setTxtProgressBar(pb, n)
}
close(pb)
t1 <- proc.time()

##==============================================================================
##==============================================================================

# Model 2 - plots
itmp <- which(jseq[,,1]==46)

plot(cite,subs,col='black',pch=16)
  points(cite[itmp],subs[itmp],col='red',pch=16)

plot(cite,tacc,col='black',pch=16)
  points(cite[itmp],tacc[itmp],col='red',pch=16)


##==============================================================================
## End
##==============================================================================
