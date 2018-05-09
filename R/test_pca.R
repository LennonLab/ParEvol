rm(list=ls())
getwd()
setwd("~/GitHub/ParEvol")

library(amap)
library(ggplot2)

pop_by_gene <- c("data/Good_et_al/gene_by_pop_delta.txt")
df <- read.table(paste(pop_by_gene, collapse = ''), sep = "\t", 
                 header = TRUE, row.names = 1)

test<-which(apply(df, 2, var)==0)
df.test <- df[ , apply(df, 2, var) != 0]

pca1 <- prcomp(df.test, scale. = TRUE)
scores <- as.data.frame(pca1$x)

plot(pca1$x[,1:2])

# plot of observations
ggplot(data = scores, aes(x = PC1, y = PC2, label = rownames(scores))) +
  geom_hline(yintercept = 0, colour = "gray65") +
  geom_vline(xintercept = 0, colour = "gray65") +
  ggtitle("PCA plot of USA States - Crime Rates")
