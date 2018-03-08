rm(list=ls())
getwd()
setwd("~/GitHub/ParEvol")

# Load dependencies
require("vegan")
require("png")
require("grid")
library("scales")


# pop-by-gene matrix
pop_by_gene <- c("data/ltee/gene_by_pop_m_I.txt")
df <- read.table(paste(pop_by_gene, collapse = ''), sep = "\t", header = TRUE, row.names = 1)
# remove rows with all zeros
df.no0 <- df[apply(df[,-1], 1, function(x) !all(x==0)),]
df.no0.db <- vegdist(df.no0, method = "bray", upper = TRUE, diag = TRUE)
df.pcoa <- cmdscale(df.no0.db, eig = TRUE, k = 2) 
explainvar1 <- round(df.pcoa$eig[1] / sum(df.pcoa$eig), 3) * 100
explainvar2 <- round(df.pcoa$eig[2] / sum(df.pcoa$eig), 3) * 100

png(filename = paste(c("figs/pcoa.png"), collapse = ''),
    width = 1200, height = 1200, res = 96*2)

par(mar = c(6.5, 6, 1.5, 2.5) + 0.1)
# Initiate Plot

plot(df.pcoa$points[ ,1], df.pcoa$points[ ,2],  xlim = c(-0.7, 0.7), ylim = c(-0.7, 0.7),
     xlab = paste("PCoA 1 (", explainvar1, "%)", sep = ""),
     ylab = paste("PCoA 2 (", explainvar2, "%)", sep = ""),
     pch = 2, cex = 2.0, type = "n", cex.lab = 1.5, cex.axis = 1.2, axes = FALSE)

# Add Axes
axis(side = 1, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
axis(side = 2, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
abline(h = 0, v = 0, lty = 3)
box(lwd = 2)


# Add Points & Labels....
# add color mutator vs non mutator
#then add time points....
cols <- c()
treats <- c()
for (x in rownames(df.pcoa.$points)){
  if (grepl("frequency_L0", x)){
    treats <- c(treats, "1")
    cols <- c(cols, "#87CEEB")
  } else if ( grepl("frequency_L1", x)) {
    treats <- c(treats, "10")
    cols <- c(cols.D100, "#FFA500")
  } else if (grepl("frequency_L2", x)) {
    treats <- c(treats, "100")
    cols <- c(cols, "#FF6347")
  }
}


#plot
points(df.pcoa$points[ ,1], df.pcoa$points[ ,2],
       pch = 1, cex = 1.5, bg = "gray", col = alpha('black', 0.5), lwd  = 3)
#ellipse.cols <- c("#87CEEB", "#FFA500", "#FF6347", "#87CEEB", "#FFA500", "#FF6347")
#ordiellipse(df.m.pcoa., test$merge, conf = 0.95, col = ellipse.cols, lwd=2)
#legend(x=-0.8,y = -1.01, xpd = TRUE, c("1-day","10-day","100-day"), 
#       col=c('#87CEEB','#FFA500','#FF6347'), ncol=3, bty ="n", 
#       pch = c(16,16,16), cex = 1.6, pt.cex = 3.0, text.font = 20)
dev.off()

# Show Plot
img <- readPNG(paste(c("figs/pcoa.png"), collapse = ''))
grid.raster(img)

# run the betadisper for each time point...
