rm(list = ls())
getwd()
setwd("~/GitHub/ParEvol/")

library(PLNmodels)
library(data.table)
#library(foreach)
#library(itertools)

df.tenaillon <- read.table("data/Tenaillon_et_al/gene_by_pop_delta.txt", 
                           header = TRUE, sep = "\t", row.names = 1)
matrix.mult <- as.matrix(df.tenaillon)

iter <- 10
#sample_sizes <- c( 2,   7,  13,  19,  25,  31,  37,  43,  49,  55,  61,  67,  73,
#                   79,  85,  91,  97, 103, 109, 115)
sample_sizes <- c(37)

get.network.samples <- function(){
  for (sample_size in sample_sizes){
    for (i in seq(1, 10, by=1)){
      print(i)
      matrix.mult.sample <- matrix.mult[sample(1:nrow(matrix.mult), sample_size, replace=FALSE),]
      matrix.mult.sample <- matrix.mult.sample[, colSums(matrix.mult.sample != 0) > 0]
      #print(rownames(matrix.mult.sample))
      mult.models <- PLNnetwork(matrix.mult.sample ~ 1)
      mult.model.BIC <- mult.models$getBestModel("BIC")
      table.name <- paste("data/Tenaillon_et_al/network_sample_size/network_sample_", sample_size, "_iter_", i,".txt", sep = "")
      write.table(as.matrix(mult.model.BIC$latent_network()), file = table.name, sep = "\t")
    }
  }
}

get.network.samples()


#sample_size <- 19
#matrix.mult.sample <- matrix.mult[sample(1:nrow(matrix.mult), sample_size, replace=FALSE),]
#matrix.mult.sample <- matrix.mult.sample[, colSums(matrix.mult.sample != 0) > 0]
#mult.models <- PLNnetwork(matrix.mult.sample ~ 1)
#mult.model.BIC <- mult.models$getBestModel("BIC")
#table.name <- paste("data/Tenaillon_et_al/network_sample_size/network_sample_", sample_size, "_iter_", "1",".txt", sep = "")
#write.table(as.matrix(mult.model.BIC$latent_network()), file = table.name, sep = "\t")

