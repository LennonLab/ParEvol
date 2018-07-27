rm(list = ls())
getwd()
setwd("~/GitHub/ParEvol/")

library(PLNmodels)
library(data.table)
library(foreach)
library(itertools)

# generate network for tenaillon dataset 
df.tenaillon <- read.table("data/Tenaillon_et_al/gene_by_pop_delta.txt", 
                 header = TRUE, sep = "\t", row.names = 1)

matrix.mult <- as.matrix(df.tenaillon)
mult.models <- PLNnetwork(matrix.mult ~ 1)
mult.model.BIC <- mult.models$getBestModel("BIC")

write.table(as.matrix(mult.model.BIC$latent_network()), file = "data/Tenaillon_et_al/network.txt", sep = "\t")

# generate network for good dataset, make a new network for each timepoint
df.good <- read.table("data/Good_et_al/gene_by_pop_delta.txt", 
                           header = TRUE, sep = "\t", row.names = 1)
# only look at nonmutators (for now)
complete_nonmutator_lines <- c('m5','m6','p1','p2','p4','p5')
complete_mutator_lines <- c('m1','m4','p3')
to_keep <- rownames(df.good) %like% "m5" + rownames(df.good) %like% "m6" + 
  rownames(df.good) %like% "p1" + rownames(df.good) %like% "p2" + 
  rownames(df.good) %like% "p4" + rownames(df.good) %like% "p5"
df.good.noMut <- df.good[as.logical(to_keep),]

get.time.network <- function(gene.by.pop){
  times <- c()
  for (x in rownames(df.good)){
    time <- strsplit(x, '_')[[1]][2]
    times <- c(times, time)
  }
  times <- unique(times)
  times <- tail(times, -2)
  for (time in times){
    time.mod <- paste('_', time, sep = "")
    to.keep <- gene.by.pop[rownames(gene.by.pop) %like% time.mod, ]
    to.keep <- to.keep[which(rowSums(to.keep) > 0), ] 
    to.keep <- to.keep[, which(colSums(to.keep) > 0)]
    to.keep.matrix.mult <- as.matrix(to.keep)
    to.keep.matrix.mult.models <- PLNnetwork(to.keep.matrix.mult ~ 1)
    #StARS
    to.keep.matrix.mult.models.BIC <- to.keep.matrix.mult.models$getBestModel("StARS")
    #to.keep.matrix.mult.models.BIC <- to.keep.matrix.mult.models$getBestModel("BIC")
    table.name <- paste("data/Good_et_al/networks/network_", time  ,".txt", sep = "")
    write.table(as.matrix(to.keep.matrix.mult.models.BIC$latent_network()), file = table.name, sep = "\t")
    
  }
}

get.time.network(df.good.noMut)

             