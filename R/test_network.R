rm(list = ls())
getwd()
setwd("~/GitHub/ParEvol/")

#library(devtools)
#devtools::install_github("jchiquet/PLNmodels")
library(PLNmodels)
library(ade4)
data("trichometeo")

dim(trichometeo$fau)
head(trichometeo$fau)

abundance <- as.matrix(trichometeo$fau) ## must be a matrix
# test covariates
test.cov <- c()
models <- PLNnetwork(abundance ~ 1)

model.BIC   <- models$getBestModel("BIC")   # if no criteria is specified, the best BIC is used
model.StARS <- models$getBestModel("StARS") # if StARS is requested, stabiltiy selection is performed if needed 
# StARS usually outperforms BIC, so use that 
test.cov <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
models.cov <- PLNnetwork(abundance ~ test.cov)
#covar <- trichometeo$meteo

model.BIC$plot_network()




# test with mutation data....
df <- read.table("data/Tenaillon_et_al/gene_by_pop_delta.txt", 
                 header = TRUE, sep = "\t", row.names = 1)
matrix.mult <- as.matrix(df)
mult.models <- PLNnetwork(matrix.mult ~ 1)
mult.models2 <- PLNnetwork(matrix.mult ~ 1)

mult.model.BIC <- mult.models$getBestModel("BIC")
mult.model2.BIC <- mult.models2$getBestModel("BIC")

#mult.model.StARS <- mult.models$getBestModel("StARS")

par(mfrow = c(1, 1))
mult.model.BIC$plot_network()

par(mfrow = c(1, 1))
mult.model2.BIC$plot_network()


write.table(as.matrix(mult.model2.BIC$latent_network()), file = "data/Tenaillon_et_al/network.txt", sep = "\t")


as.matrix(mult.model2.BIC$latent_network())[,"kpsD"]


# PLNPCA
matrix.mult.PLNPCA <- PLNPCA(matrix.mult ~ 1, ranks=1:5)
matrix.mult.PLNPCA$plot()
matrix.mult.PLNPCA.ICL <- matrix.mult.PLNPCA$getBestModel("ICL")
matrix.mult.PLNPCA.ICL$plot_PCA()

#matrix.mult.PLNPCA.ICL$scores
par(mar = c(1, 5, 2, 2) + 0.1)
plot(matrix.mult.PLNPCA.ICL$scores[,1], matrix.mult.PLNPCA.ICL$scores[,2])
points(matrix.mult.PLNPCA.ICL$scores[,1], matrix.mult.PLNPCA.ICL$scores[,2])



