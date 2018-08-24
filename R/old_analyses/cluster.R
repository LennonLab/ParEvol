require("cluster")
comm.bc.clust <- hclust(df.no0.db, method = "average")
comm.bc.cut <- cutree(comm.bc.clust, k=3)

# compute gap statistic
set.seed(123)
gap_stat <- clusGap(df.pcoa$points, FUN = kmeans, K.max = 6, B = 50, 
                    spaceH0 = 'scaledPCA')


#fviz_nbclust(df.pcoa$points, FUNcluster = hcut, method = 'gap_stat', k.max = 6)
fviz_nbclust(df.pcoa$points, kmeans, method = "silhouette")

fviz_gap_stat(gap_stat)




kfit <- kmeans(df.no0.db, 3)

get.symbol <- function(pcoa.points){
  pop.symbols <- c()
  for (x in rownames(pcoa.points)){
    pop <- toString(strsplit(x, '_')[[1]][1])
    if (grepl("m5", pop)){
      # square
      pop.symbol <- 0
    } else if ( grepl("m6", pop)) {
      pop.symbol <- 1
      # circle
    } else if ( grepl("p1", pop)) {
      pop.symbol <- 2
      # triangle
    } else if ( grepl("p2", pop)) {
      pop.symbol <- 3
      # addition sign
    } else if ( grepl("p4", pop)) {
      pop.symbol <- 4
      # multiplication sign
    } else if ( grepl("p5", pop)) {
      # diamond
      pop.symbol <- 5
    } 
    pop.symbols <- c(pop.symbols, pop.symbol)
  }
  return(pop.symbols)
}
#kfit$cluster
pcoa.k1 <- df.pcoa$points[comm.bc.cut == 1,]
pcoa.k2 <- df.pcoa$points[comm.bc.cut == 2,]
pcoa.k3 <- df.pcoa$points[comm.bc.cut == 3,]

lightcolours <- c("darkolivegreen3", "cadetblue3", 'red')

png(filename = paste(c("figs/pcoa_clust.png"), collapse = ''),
    width = 1200, height = 900, res = 96*2)
par(mar = c(5, 7, 1, 2) + 0.1)
# plots for each population
pcoa.clust.plot <- plot(df.pcoa$points[ ,1], df.pcoa$points[ ,2],
                        xlim = c(-0.7, 0.7), ylim = c(-0.7, 0.7), 
                        cex = 2.0, type = "n", cex.lab = 1.5, cex.axis = 1.2, 
                        axes = FALSE, xlab = '', ylab ='')
points(pcoa.k1[ ,1], pcoa.k1[ ,2],
       pch= get.symbol(pcoa.k1), cex = 1.5, bg = "gray", col = alpha("darkolivegreen3", 0.5), lwd  = 3)
points(pcoa.k2[ ,1], pcoa.k2[ ,2],
       pch= get.symbol(pcoa.k2), cex = 1.5, bg = "gray", col = alpha("cadetblue3", 0.5), lwd  = 3)
points(pcoa.k3[ ,1], pcoa.k3[ ,2],
       pch= get.symbol(pcoa.k3),cex = 1.5, bg = "gray", col = alpha("red", 0.5), lwd  = 3)
# Add Axes
axis(side = 1, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
axis(side = 2, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
abline(h = 0, v = 0, lty = 3)
box(lwd = 2)

mtext(paste("PCoA 1 (", explainvar1, "%)", sep = ""), side=1, line=1, 
      cex=1.5, col="black", outer=TRUE)  
mtext(paste("PCoA 2 (", explainvar2, "%)", sep = ""), side=2, line=1, 
      cex=1.5, col="black", outer=TRUE)

dev.off()

# Show Plot
img <- readPNG(paste(c("figs/pcoa_clust.png"), collapse = ''))
grid.raster(img)