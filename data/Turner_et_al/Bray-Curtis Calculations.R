library(reshape2)
library(vegan)
library(gtools)
library(combinat)

#read in list of mutations as a dataframe
mut_list <- read.csv("Breseq_Output_with_verification.txt", header=TRUE, sep="\t" )
#reformat mutation frequency as a numeric value
mut_list$Frequency <- as.numeric(sub("%", "",mut_list$Frequency,fixed=TRUE))/100

#test for treatment differences in # of mutations
mut_list_real <- subset(mut_list, Real.=="Y")
treat_list <- c(rep("LBHC",5), rep("SBHC",6), rep ("LBLC",6), rep("HC",6), rep("LC",6))
mutation_count_real <- table(mut_list_real$Sample)
min(mutation_count_real)
max(mutation_count_real)
tapply(as.vector(mutation_count_real),treat_list, mean)
kruskal.test(mutation_count_real, as.factor(treat_list))

#include only validated mutations
mut_list$BC.Include. <- toupper(mut_list$BC.Include.)
mut_list_valid <- subset(mut_list, BC.Include.=="Y")

#reshape the mutation list into a community data matrix
community <- dcast(mut_list_valid, Sample~Gene.Assigned,value.var="Frequency",sum)
comm <- community[,-1]
rownames(comm) <- community[,1] 

#calculate bray curtis similarity and convert to a data frame
braycurtis <- vegdist(comm)
bc_matrix <- as.matrix(braycurtis)
bcsim_matrix <- (1-bc_matrix)
write.csv(bcsim_matrix, file="braycurtis_similarity_matrix.csv")

treat_list <- c(rep("LBHC",5), rep("SBHC",6), rep ("LBLC",6), rep("HC",6), rep("LC",6))
rownames(bcsim_matrix) <- treat_list
colnames(bcsim_matrix) <-treat_list
braycurtis_frame <- as.data.frame(as.matrix(braycurtis))


#functions for calculating mean Bray-Curtis similarity in subsets of the data

bc_within <- function(within){
  #returns mean within treatment values in a matrix
  sum <- 0
  count <- 0
  for(i in 1:(nrow(within)-1)){
    for(j in (i+1):ncol(within)){
       if(rownames(within)[i] == colnames(within)[j]){
          sum <- sum+within[i,j]
         count <- count+1
      }
    }
  }
  return(sum/count)
}

bc_between <- function(between){
  #returns mean between treatment values in a matrix
  sum <- 0
  count <- 0
  for(i in 1:(nrow(between)-1)){
    for(j in (i+1):ncol(between)){
      if(rownames(between)[i] != colnames(between)[j]){
        sum <- sum+between[i,j]
        count <- count+1
      }
    }
  }
  return(sum/count)
}

bc_similarity <- function(similar){
  #returns difference between mean of similar and non-similar (different) treatments
  sum_sim <- 0
  count_sim <- 0
  sum_diff <- 0
  count_diff <- 0
  for(i in 1:(nrow(similar)-1)){
    for(j in (i+1):ncol(similar)){
      if(rownames(similar)[i] != colnames(similar)[j]){
        if((rownames(similar)[i]==("LBHC")||rownames(similar)[i]=="SBHC") && colnames(similar)[j]=="LC"){
          sum_diff <- sum_diff+similar[i,j]
          count_diff <- count_diff+1
        } else if(rownames(similar)[i]==("LBLC") && colnames(similar)[j]=="HC"){
          sum_diff <- sum_diff+similar[i,j]
          count_diff <- count_diff+1
        } else if(rownames(similar)[i]==("HC") && colnames(similar)[j]=="LBLC"){
          sum_diff <- sum_diff+similar[i,j]
          count_diff <- count_diff+1
        } else if(rownames(similar)[i]==("LC") && (colnames(similar)[j]=="LBHC"||colnames(similar)[j]=="SBHC")){
          sum_diff <- sum_diff+similar[i,j]
          count_diff <- count_diff+1
          
        } else {
          sum_sim <- sum_sim+similar[i,j]
          count_sim <- count_sim+1
        }
      }
    }
  }
  return(sum_sim/count_sim-sum_diff/count_diff)
}

bc_within(bcsim_matrix)
bc_between(bcsim_matrix)
bc_similarity(bcsim_matrix)


#Testing whether bray-curtis values are more similar within treatments than between treatments
#and whether similarity is higher in environments with shared characteristics than environments without shared characteristics

difference <- bc_within(bcsim_matrix)-bc_between(bcsim_matrix)

difference_rand <- character(length(10^5))
envir_similarity <- character(length(10^5))
randomizing_matrix <-bcsim_matrix

for(i in 1:10^5)
{
  treat_list_random <- sample(treat_list)
  rownames(randomizing_matrix) <- treat_list_random
  colnames(randomizing_matrix) <- treat_list_random
  between <- bc_between(randomizing_matrix)
  within <- bc_within(randomizing_matrix)
  similarity <- bc_similarity(randomizing_matrix)
  difference_rand[i] <- (between-within)
  envir_similarity[i] <- similarity
}

#probability of getting a larger or equal difference between within-treatment bray-curtis and between-treatment bray-curtis by chance
mean(as.numeric(difference_rand)>=difference)

#probability of getting a larger or equal difference between similary-treatment bray-curtis and dissimilar-treatment bray-curtis by chance
mean(as.numeric(envir_similarity)>=bc_similarity(bcsim_matrix))


#functions for permutation tests

upermn <- function(x) {
  #calculates number of unique permutations of x
  #copied from http://stackoverflow.com/questions/5671149/permute-all-unique-enumerations-of-a-vector-in-r
  n <- length(x)
  duplicates <- as.numeric(table(x))
  factorial(n) / prod(factorial(duplicates))
}

uperm <- function(x) {
  #returns a list of all the unique permutations of the elements of vector x
  # copied from http://stackoverflow.com/questions/5671149/permute-all-unique-enumerations-of-a-vector-in-r
  u <- sort(unique(x))
  l <- length(u)
  if (l == length(x)) {
    return(do.call(rbind,permn(x)))
  }
  if (l == 1) return(x)
  result <- matrix(NA, upermn(x), length(x))
  index <- 1
  for (i in 1:l) {
    v <- x[-which(x==u[i])[1]]
    newindex <- upermn(v)
    if (table(x)[i] == 1) {
      result[index:(index+newindex-1),] <- cbind(u[i], do.call(rbind, unique(permn(v))))
    } else {
      result[index:(index+newindex-1),] <- cbind(u[i], uperm(v))
    }
    index <- index+newindex
  }
  return(result)
}

permutation_bc <- function(x){
  #returns bc_within-bc_between for all permutations of treatment assignments for a pair of treatments
  p <- uperm(rownames(x))
  temp <- x
  diff_rand <- numeric(length=nrow(p))
  for(row in 1:nrow(p)){
    rownames(temp) <- p[row,]
    colnames(temp) <- p[row,]
    diff_rand[row] <- (bc_within(temp) - bc_between(temp))
  }
  return(diff_rand)
}

#Testing whether pairs of treatments differ significantly more between treatments than within treatments

# Loop through each treatment pair
treatments <- c("LBHC","SBHC","LBLC","HC", "LC")
for(m in 1:(length(treatments)-1)){
  for(n in (m+1):length(treatments)){
    a <- treatments[m]
    b <- treatments[n]

    #make a matrix which only includes 2 selested treatments
    include_list <- c(a,b)
     subcomm <- bcsim_matrix[include_list,include_list]
    subcommunity<- bcsim_matrix[(rownames(bcsim_matrix)== a)|(rownames(bcsim_matrix)== b),(colnames(bcsim_matrix)==a | colnames(bcsim_matrix)==b)]
    
    #Calculate difference between between and within treatment bray curtis
    between_ab <- bc_between(subcommunity)
    within_ab <- bc_within(subcommunity)
    #For all permutations of treatment assignemnts, calculate within-between
    permutation_diff <- permutation_bc(subcommunity)
    #Calculate significance
    significance <- mean(as.numeric(permutation_diff)>=(within_ab-between_ab))
    #Record outcome
    print(c(a,b,significance))
  }
}  








