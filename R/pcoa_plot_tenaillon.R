rm(list=ls())
getwd()
setwd("~/GitHub/ParEvol")

require("vegan")
require("png")
require("grid")
library("scales")

pop_by_gene <- c("data/Tenaillon_et_al/gene_by_pop_delta.txt")
df <- read.table(paste(pop_by_gene, collapse = ''), sep = "\t", 
                 header = TRUE, row.names = 1)

df.db <- vegdist(df, method = "bray", upper = TRUE, diag = TRUE)
df.pcoa <- cmdscale(df.db, eig = TRUE, k = 8) 
explainvar1 <- round(df.pcoa$eig[1] / sum(df.pcoa$eig), 3) * 100
explainvar2 <- round(df.pcoa$eig[2] / sum(df.pcoa$eig), 3) * 100


png(filename = paste(c("figs/pcoa_test_Tenaillon.png"), collapse = ''),
    width = 1200, height = 900, res = 96*2)

par(mar = c(5, 5, 1, 5) + 0.1, xpd=T)
# Initiate Plot
plot(df.pcoa$points[ ,1], df.pcoa$points[ ,2],  xlim = c(-0.7, 0.7), 
     ylim = c(-0.7, 0.7),
     xlab = paste("PCoA 1 (", explainvar1, "%)", sep = ""),
     ylab = paste("PCoA 2 (", explainvar2, "%)", sep = ""),
     pch = 2, cex = 2.0, type = "n", cex.lab = 1.5, 
     cex.axis = 1.2, axes = FALSE)

points(df.pcoa$points[ ,1], df.pcoa$points[ ,2],
       pch = 1, cex = 1.5, bg = "gray", col = alpha('darkgreen', 0.5), lwd  = 3)
points(0, 0, pch = 16, cex = 2.5, bg = 'gray', col = 'gray', lwd = 1.5)

# Add Axes
axis(side = 1, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
axis(side = 2, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
clip(-0.8,0.8, -0.8,0.8)
abline(h = 0, v = 0, lty = 3)
box(lwd = 2)

dev.off()
# Show Plot
img <- readPNG(paste(c("figs/pcoa_test_Tenaillon.png"), collapse = ''))
grid.raster(img)
