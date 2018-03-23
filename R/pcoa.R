rm(list=ls())
#dev.off()
getwd()
setwd("~/GitHub/ParEvol")

# Load dependencies
require("vegan")
require("png")
require("grid")
library("scales")
library("data.table")

# pop-by-gene matrix
pop_by_gene <- c("data/ltee/gene_by_pop_m_I.txt")
df <- read.table(paste(pop_by_gene, collapse = ''), sep = "\t", header = TRUE, row.names = 1)
# complete_lines = ['m1', 'm4', 'm5', 'm6', 'p1', 'p2', 'p3', 'p4', 'p5']    
# complete_nonmutator_lines = ['m5','m6','p1','p2','p4','p5']
# complete_mutator_lines = ['m1','m4','p3']
# complete_early_mutator_lines = ['m4','p3']
# early_mutator_lines = ['m4','p3']
complete_nonmutator_lines <- c('m5','m6','p1','p2','p4','p5')
complete_mutator_lines <- c('m1','m4','p3')
to_keep <- rownames(df) %like% "m5" + rownames(df) %like% "m6" + 
  rownames(df) %like% "p1" + rownames(df) %like% "p2" + 
  rownames(df) %like% "p4" + rownames(df) %like% "p5" #+
  #rownames(df) %like% "m1" + rownames(df) %like% "m4" +
  #rownames(df) %like% "p3"

df.noMut <- df[as.logical(to_keep),]
#df.noMut <- df
# remove rows with all zeros
df.no0 <- df.noMut[apply(df.noMut[,-1], 1, function(x) !all(x==0)),]
df.no0.db <- vegdist(df.no0, method = "bray", upper = TRUE, diag = TRUE)
df.pcoa <- cmdscale(df.no0.db, eig = TRUE, k = 3) 
explainvar1 <- round(df.pcoa$eig[1] / sum(df.pcoa$eig), 3) * 100
explainvar2 <- round(df.pcoa$eig[2] / sum(df.pcoa$eig), 3) * 100

times <- c()
pops <- c()
for (x in rownames(df.pcoa$points)){
  time <- strsplit(x, '_')[[1]][2]
  times <- c(times, time)
}
# get colors for times
times.sorted <- as.character(sort(as.numeric(unique(times))))
number.times <- length(times.sorted)
colfunc.nonMut <- colorRampPalette(c("lightgreen", "darkgreen"))
colfunc.Mut <- colorRampPalette(c("indianred1", "darkred"))

time.cols.nonMut <- colfunc.nonMut(number.times)
time.cols.Mut <- colfunc.Mut(number.times)

times.cols.pcoa <- c()
pop.symbols <- c()
for (x in rownames(df.pcoa$points)){
  time <- strsplit(x, '_')[[1]][2]
  pop <- toString(strsplit(x, '_')[[1]][1])
  time.position <- match(time, times.sorted)
  if (pop %in% complete_nonmutator_lines){
    time.color <- time.cols.nonMut[time.position]
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
  } 
  else if (pop %in% complete_mutator_lines) {
    time.color <- time.cols.Mut[time.position]
    if ( grepl("m1", pop)) {
      # diamond
      pop.symbol <- 7
    } 
    else if ( grepl("m4", pop)) {
      # diamond
      pop.symbol <- 9
    } 
    else if ( grepl("p3", pop)) {
      # diamond
      pop.symbol <- 10
    } 
  }
  pop.symbols <- c(pop.symbols, pop.symbol)
  times.cols.pcoa <- c(times.cols.pcoa, time.color)
}



# get symbols for each population
#pop.symbols <- c()
#for (x in rownames(df.pcoa$points)){
#  pop <- strsplit(x, '_')[[1]][1]
#  else if ( grepl("m1", pop)) {
#    # diamond
#    pop.symbols <- c(pop.symbols, 7)
#  } 
#  else if ( grepl("m4", pop)) {
#    # diamond
#    pop.symbols <- c(pop.symbols, 9)
#  } 
#  else if ( grepl("p3", pop)) {
#    # diamond
#    pop.symbols <- c(pop.symbols, 10)
#  } 
#}

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



#plot
points(df.pcoa$points[ ,1], df.pcoa$points[ ,2],
       pch = pop.symbols, cex = 1.5, bg = "gray", col = alpha(times.cols.pcoa, 0.5), lwd  = 3)
dev.off()

# Show Plot
img <- readPNG(paste(c("figs/pcoa.png"), collapse = ''))
grid.raster(img)

# run the betadisper for each time point...

# get euclidean distance
# get the time steps
time_steps <- c()
for (index in seq(1, length(head(times.sorted, -1)))) {
  time_step <- paste(times.sorted[index], times.sorted[index+1], sep = "_")
  time_steps <- c(time_steps, time_step)
}

euc.df <- as.data.frame(matrix(data=NA,nrow=length(complete_nonmutator_lines), ncol=length(tail(times.sorted, -1))))

rownames(euc.df) <- complete_nonmutator_lines
#colnames(euc.df) <- time_steps
colnames(euc.df) <- tail(times.sorted, -1)

for (pop in complete_nonmutator_lines) {
  for (time_step in time_steps){
    time1 <- strsplit(time_step, '_')[[1]][1]
    time2 <- strsplit(time_step, '_')[[1]][2]
    pop.time1 <- paste(pop, time1, sep = "_")
    pop.time2 <- paste(pop, time2, sep = "_")
    if ( (pop.time1 %in% rownames(df.pcoa$points)) & (pop.time2 %in% rownames(df.pcoa$points)) ) {
      #print(pop.time1)
      #print(df.pcoa$points[pop.time1, ]) 
      euc.dist <- dist(rbind(df.pcoa$points[pop.time1, ], df.pcoa$points[pop.time2, ]))[1]
      #print(rbind(df.pcoa$points[pop.time1, ], df.pcoa$points[pop.time2, ]))
      euc.df[pop, time2] <- euc.dist
    }
  } 
}

euc.df.m5 <- euc.df['m5',] 
euc.df.m5.clean <- euc.df.m5[,!is.infinite(colSums(euc.df.m5)) 
                                         & !is.na(colSums(euc.df.m5)) 
                             & (log10(colSums(euc.df.m5)) > -10)]
euc.df.m6 <- euc.df['m6',] 
euc.df.m6.clean <- euc.df.m6[,!is.infinite(colSums(euc.df.m6)) 
                                         &  !is.na(colSums(euc.df.m6)) 
                                   & (log10(colSums(euc.df.m6)) > -10) ]
euc.df.p1 <- euc.df['p1',] 
euc.df.p1.clean <- euc.df.p1[,!is.infinite(colSums(euc.df.p1)) 
                                         &  !is.na(colSums(euc.df.p1)) 
                                   & (log10(colSums(euc.df.p1)) > -10) ]
euc.df.p2 <- euc.df['p2',] 
euc.df.p2.clean <- euc.df.p2[,!is.infinite(colSums(euc.df.p2)) 
                                         &  !is.na(colSums(euc.df.p2)) 
                                   & (log10(colSums(euc.df.p2)) > -10) ]
euc.df.p4 <- euc.df['p4',] 
euc.df.p4.clean <- euc.df.p4[,!is.infinite(colSums(euc.df.p4)) 
                                         &  !is.na(colSums(euc.df.p4)) 
                                   & (log10(colSums(euc.df.p4)) > -10) ]
euc.df.p5 <- euc.df['p5',] 
euc.df.p5.clean <- euc.df.p5[,!is.infinite(colSums(euc.df.p5)) 
                                         &  !is.na(colSums(euc.df.p5)) 
                             & (log10(colSums(euc.df.p5)) > -10) ]


plot(colnames(euc.df.m5.clean), euc.df.m5.clean)









