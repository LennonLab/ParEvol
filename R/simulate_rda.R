rm(list=ls())
getwd()
setwd("~/GitHub/ParEvol")

library(vegan)
library(plyr)

set.seed(123456)

get_pop_matrix <- function(n_pops, n_genes, n_muts, probs, env){
  out.df <- as.data.frame(matrix(NA, nrow=0, ncol=n_genes))
  colnames(out.df) <- seq(1, n_genes, by=1)
  for (i in 1:n_pops){
    sample_i <- sample(n_genes, size=n_muts, replace=TRUE, prob = probs)
    sample_i.df <- as.data.frame(t(as.matrix(table(sample_i))))
    #out.matrix <- rbind(out.matrix, sample_i.matrix)
    out.df <- rbind.fill(out.df, sample_i.df)
  }
  # code for rownames
  # fill NA with zeros
  out.df[is.na(out.df)] <- 0
  rownames(out.df) <- paste('env', env, '_', rownames(out.df), sep = "")   
  return(out.df)
}


run_simulation <- function(n_genes, n_muts_array, n_pops_array, gamma_shapes, gamma_scales, iter, file_name, perm.num=10000){
  sink( paste("./data/rda_sims/" , file_name, ".txt", sep=""))
  out.header <- paste("N_genes", "N_genes_sample", "N_muts", "N_pops", "Gamme_shape", "Gamma_scale", "Iteration", "F_null", "p", sep="\t")
  cat(out.header)
  cat("\n")
  gene_sample_sizes <- seq(2, n_genes, by=2)
  #gene_sample_size <- gene_sample_sizes[3]
  #gene_sample_size <- 1
  for (gene_sample_size in gene_sample_sizes){
    for (n_muts_i in n_muts_array){
      for (n_pops_i in n_pops_array){
        #print(gene_sample_size, n_muts_i, n_pops_i)
        for (gamma_shape_i in gamma_shapes){
          for (gamma_scale_i in gamma_scales){
            for (i in 1:iter){
              sample_rates <- rgamma (n=n_genes, shape=gamma_shape_i, scale=gamma_scale_i)
              gene_sample <- sample(n_genes, size = gene_sample_size, replace=FALSE)
              to_reshuffle <- sample_rates[c(gene_sample)]
              sample_rates_no_reshuffle <- sample_rates[-c(gene_sample)]
              
              sample_rates_env1 <- append(sample_rates_no_reshuffle, 
                                          sample(to_reshuffle, size=length(to_reshuffle), replace = FALSE))
              sample_rates_env1_prob <- sample_rates_env1 / sum(sample_rates_env1)
              sample_rates_env2 <- append(sample_rates_no_reshuffle, 
                                          sample(to_reshuffle, size=length(to_reshuffle), replace = FALSE))
              sample_rates_env2_prob <- sample_rates_env2 / sum(sample_rates_env2)
              
              mutation_counts_env1<-get_pop_matrix(n_pops=n_pops_i, n_genes=n_genes, 
                                                   n_muts=n_muts_i, probs=sample_rates_env1_prob, 
                                                   env='1')
              mutation_counts_env2<-get_pop_matrix(n_pops=n_pops_i, n_genes=n_genes, 
                                                   n_muts=n_muts_i, probs=sample_rates_env2_prob, 
                                                   env='2')
  
              mutation_counts <- rbind(mutation_counts_env1, mutation_counts_env2)
              # Hellinger transform the data
              mutation_counts.H <- decostand(mutation_counts, method='hellinger')
              envs <- as.character(lapply(strsplit(rownames(mutation_counts) , "_"), function(x) x[1]))
              rda.envs <- rda(mutation_counts ~ envs, each=10)
              rda.envs.perm <- permutest(rda.envs, permutations=perm.num, model='full')
              F.value <- as.numeric(rda.envs.perm$F.0)
              p.score <- sum( as.numeric(c(rda.envs.perm$F.perm)) > F.value) / perm.num
              
              out.line <- paste(as.character(n_genes),
                                as.character(gene_sample_size),
                                as.character(n_muts_i),
                                as.character(n_pops_i),
                                as.character(gamma_shape_i),
                                as.character(gamma_scale_i),
                                as.character(i), 
                                as.character(F.value), 
                                as.character(p.score), sep="\t")
              cat(out.line)
              cat("\n")
            }
          }
        }
      }
    }
  }
  sink()
}

n_genes <- 50
gamma_shapes <- c(1)
gamma_scales <- c(3)
iter <- 1000

# start with 5 and 50 for muts, then5 and 50 for N
n_pops_array1 <- seq(4, 50, by=2)
n_muts_array1 <-c(5)
file_name1 <- "rda_pops_mut5_overlap"
# include iteration code, loop through number of genes
run_simulation(n_genes=n_genes, 
               n_muts_array=n_muts_array1, 
               n_pops_array=n_pops_array1,
               gamma_shapes=gamma_shapes, 
               gamma_scales=gamma_scales,
               file_name=file_name1,
               iter=iter)


n_pops_array2 <-seq(4, 50, by=2)
n_muts_array2 <- c(50) 
file_name2 <- "rda_pops_mut50_overlap"
run_simulation(n_genes=n_genes, 
               n_muts_array=n_muts_array2, 
               n_pops_array=n_pops_array2,
               gamma_shapes=gamma_shapes, 
               gamma_scales=gamma_scales,
               file_name=file_name2,
               iter=iter)


n_pops_array3 <-  c(5)
n_muts_array3 <- seq(4, 50, by=2)
file_name3 <- "rda_muts_pop5_overlap"
run_simulation(n_genes=n_genes, 
               n_muts_array=n_muts_array3, 
               n_pops_array=n_pops_array3,
               gamma_shapes=gamma_shapes, 
               gamma_scales=gamma_scales,
               file_name=file_name3,
               iter=iter)


n_pops_array4 <-  c(50)
n_muts_array4 <- seq(4, 50, by=2)
file_name4 <- "rda_muts_pop50_overlap"
run_simulation(n_genes=n_genes, 
               n_muts_array=n_muts_array4, 
               n_pops_array=n_pops_array4,
               gamma_shapes=gamma_shapes, 
               gamma_scales=gamma_scales,
               file_name=file_name4,
               iter=iter)



