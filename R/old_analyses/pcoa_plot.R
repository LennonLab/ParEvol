rm(list=ls())
getwd()
setwd("~/GitHub/ParEvol")

require("vegan")
require("png")
require("grid")
#library("scales")
#library("data.table")
#library("BiodiversityR")

pop_by_gene <- c("data/ltee/gene_by_pop_delta.txt")
df <- read.table(paste(pop_by_gene, collapse = ''), sep = "\t", 
                 header = TRUE, row.names = 1)
# only look at nonmutators (for now)
complete_nonmutator_lines <- c('m5','m6','p1','p2','p4','p5')
complete_mutator_lines <- c('m1','m4','p3')
to_keep <- rownames(df) %like% "m5" + rownames(df) %like% "m6" + 
  rownames(df) %like% "p1" + rownames(df) %like% "p2" + 
  rownames(df) %like% "p4" + rownames(df) %like% "p5"
df.noMut <- df[as.logical(to_keep),]
df.no0 <- df.noMut[apply(df.noMut[,-1], 1, function(x) !all(x==0)),]

df.no0.db <- vegdist(df.no0, method = "bray", upper = TRUE, diag = TRUE)
df.pcoa <- cmdscale(df.no0.db, eig = TRUE, k = 8) 
explainvar1 <- round(df.pcoa$eig[1] / sum(df.pcoa$eig), 3) * 100
explainvar2 <- round(df.pcoa$eig[2] / sum(df.pcoa$eig), 3) * 100

times <- c()
for (x in rownames(df.pcoa$points)){
  time <- strsplit(x, '_')[[1]][2]
  times <- c(times, time)
}

# function to return color gradient for time points
get.times.cols <- function(pcoa.points, times){
  # get colors for times
  times.sorted <- as.character(sort(as.numeric(unique(times))))
  number.times <- length(times.sorted)
  colfunc.nonMut <- colorRampPalette(c("lightgreen", "darkgreen"))
  time.cols.nonMut <- colfunc.nonMut(number.times)
  times.cols.pcoa <- c()
  for (x in rownames(pcoa.points)){
    time <- strsplit(x, '_')[[1]][2]
    pop <- toString(strsplit(x, '_')[[1]][1])
    time.position <- match(time, times.sorted)
    if (pop %in% complete_nonmutator_lines){
      time.color <- time.cols.nonMut[time.position]
    }
    times.cols.pcoa <- c(times.cols.pcoa, time.color)
  }
  return(times.cols.pcoa)
}



# make plot
png(filename = paste(c("figs/pcoa_test.png"), collapse = ''),
    width = 1200, height = 900, res = 96*2)
par(mfrow = c(2, 3),   
    oma = c(5, 4, 0, 0), # two rows of text at the outer left and bottom margin
    mar = c(1.5, 3.1, 1.1, 0.5), # space for one row of text at ticks and to separate plots
    pty="s") # make the plots square


pops <- c('p2', 'p4', 'm6', 'p1', 'm5', 'p5')
pop.names <- c('Ara+2', 'Ara+4', 'Ara-6', 'Ara+1', 'Ara-5', 'Ara+5')
pop.points <- c(0, 2, 4, 1, 3, 5)
  
for (i in seq_along(pops)){
  pcoa.pop <- df.pcoa$points[rownames(df.pcoa$points) %like% pops[i], ] 
  cols.pop <- get.times.cols(pcoa.pop, times)
  
  pcoa.pop.plot <- plot(pcoa.pop[ ,1], pcoa.pop[ ,2],
                       xlim = c(-0.7, 0.7), ylim = c(-0.7, 0.7), 
                       pch = 0, cex = 2.0, type = "n", cex.lab = 1.5, cex.axis = 1.2, 
                       axes = FALSE, xlab = '', ylab ='', main = pop.names[i])
  abline(h = 0, v = 0, lty = 3)
  points(pcoa.pop[ ,1], pcoa.pop[ ,2],
         pch = pop.points[i], cex = 1.5, bg = "gray", col = alpha(cols.pop, 0.5), lwd  = 3)
  points(0, 0, pch = 16, cex = 2.5, bg = 'gray', col = 'gray', lwd = 1.5)
  # Add Axes
  axis(side = 1, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
  axis(side = 2, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
  box(lwd = 2)
}

mtext(paste("PCoA 1 (", explainvar1, "%)", sep = ""), side=1, line=1, 
      cex=1.5, col="black", outer=TRUE)  
mtext(paste("PCoA 2 (", explainvar2, "%)", sep = ""), side=2, line=1, 
      cex=1.5, col="black", outer=TRUE)

dev.off()

# Show Plot
img <- readPNG(paste(c("figs/pcoa_test.png"), collapse = ''))
grid.raster(img)




# Define Plot Parameters
par(mar = c(5, 5, 1, 2) + 0.1)
# Plot Eigenvalues
plot(df.pcoa$eig, xlab = "PCoA Axis", ylab = "Eigenvalue",
     las = 1, cex.lab = 1.5, pch = 16)
# Add Expectation based on Kaiser-Guttman criterion and Broken Stick Model
abline(h = mean(df.pcoa$eig), lty = 2, lwd = 2, col = "blue")
b.stick <- bstick(29, sum(df.pcoa$eig))
lines(1:29, b.stick, type = "l", lty = 4, lwd = 2, col = "red")
# Add Legend
legend("topright", legend = c("Avg Eigenvalue", "Broken-Stick"),
       lty = c(2, 4), bty = "n", col = c("blue", "red"))





pcoa.first2 <- as.data.frame(df.pcoa$points[,1:2])
colnames(pcoa.first2)[1] <- 'axis1'
colnames(pcoa.first2)[2] <- 'axis2'
split.rows <- do.call(rbind, strsplit(rownames(pcoa.first2), '_'))
pcoa.first2$pop <- split.rows[,1]
pcoa.first2$time <- split.rows[,2]

get.symbol <- function(pcoa.points){
  pop.symbols <- c()
  for (x in rownames(pcoa.points)){
    pop <- toString(strsplit(x, '_')[[1]][1])
    if (grepl("m5", pop)){
      # square
      pop.symbol <- 3
    } else if ( grepl("m6", pop)) {
      pop.symbol <- 4
      # circle
    } else if ( grepl("p1", pop)) {
      pop.symbol <- 1
      # triangle
    } else if ( grepl("p2", pop)) {
      pop.symbol <- 0
      # addition sign
    } else if ( grepl("p4", pop)) {
      pop.symbol <- 2
      # multiplication sign
    } else if ( grepl("p5", pop)) {
      # diamond
      pop.symbol <- 5
    } 
    pop.symbols <- c(pop.symbols, pop.symbol)
  }
  return(pop.symbols)
}


png(filename = paste(c("figs/time_pcoa1.png"), collapse = ''),
    width = 1200, height = 900, res = 96*2)
#par(mar = c(5, 5, 1, 2) + 0.1)
par(mar = c(5, 5, 1, 5) + 0.1, xpd=TRUE)
plot.pcoa1 <- plot(pcoa.first2$time, pcoa.first2$axis1, 
                   xlim = c(0, 63000), 
                   ylim = c(min(pcoa.first2$axis1), max(pcoa.first2$axis1)),
                   pch = 2, cex = 2.0, type = "n", 
                   cex.lab = 1.5, cex.axis = 1.2, 
                   axes = FALSE, xlab = '', 
                   ylab ='')
points(pcoa.first2$time, pcoa.first2$axis1,cex = 1, 
       bg = "gray",  lwd  = 1, 
       pch = get.symbol(pcoa.first2), col = alpha("darkolivegreen4", 0.8))

axis(side = 1, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
axis(side = 2, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
mtext("Generations", side=1, line=3.5, cex=1.3, outer=FALSE)
mtext(paste("PCoA 1 (", explainvar1, "%)", sep = ""),
      side=2, line=3.5, cex=1.3, outer=FALSE)
box(lwd = 2)
legend("right", inset=c(-0.22,0), 
       legend=c("Ara+2","Ara+4", "Ara-6", "Ara+1", "Ara-5", "Ara+5"), 
       pch=c(0,2,4,1,3,5), title="Pop.", col="darkolivegreen4")
dev.off()
# Show Plot
img <- readPNG(paste(c("figs/time_pcoa1.png"), collapse = ''))
grid.raster(img) 






png(filename = paste(c("figs/time_pcoa2.png"), collapse = ''),
    width = 1200, height = 900, res = 96*2)
par(mar = c(5, 5, 1, 5) + 0.1, xpd=TRUE)
plot.pcoa2 <- plot(pcoa.first2$time, pcoa.first2$axis2, 
                   xlim = c(0, 63000), 
                   ylim = c(min(pcoa.first2$axis2), max(pcoa.first2$axis2)),
                   pch = 2, cex = 2.0, type = "n", 
                   cex.lab = 1.5, cex.axis = 1.2, 
                   axes = FALSE, xlab = '', 
                   ylab ='')
points(pcoa.first2$time, pcoa.first2$axis2,cex = 1, 
       bg = "gray",  lwd  = 1, 
       pch = get.symbol(pcoa.first2), col = alpha("darkolivegreen4", 0.8))

axis(side = 1, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
axis(side = 2, labels = T, lwd.ticks = 2, cex.axis = 1.2, las = 1)
mtext("Generations", side=1, line=3.5, cex=1.3, outer=FALSE)
mtext(paste("PCoA 2 (", explainvar2, "%)", sep = ""),
      side=2, line=3.5, cex=1.3, outer=FALSE)
box(lwd = 2)
legend("right", inset=c(-0.22,0), 
       legend=c("Ara+2","Ara+4", "Ara-6", "Ara+1", "Ara-5", "Ara+5"), 
       pch=c(0,2,4,1,3,5), title="Pop.", col="darkolivegreen4")
dev.off()
# Show Plot
img <- readPNG(paste(c("figs/time_pcoa2.png"), collapse = ''))
grid.raster(img)