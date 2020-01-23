#include <stdlib.h>

double similarity_undirected(unsigned short *m,unsigned short *n,size_t ncol,size_t nrow,size_t e);
double similarity(unsigned short *m,unsigned short *n,size_t ncol,size_t nrow,size_t e );
size_t analysis_ex(unsigned short *incidence,size_t ncol, size_t nrow,double *scores,size_t step,size_t max_iter,size_t verbose,size_t MAXITER,unsigned int seed);
size_t rewire_bipartite_ex(unsigned short *matrix,size_t ncol, size_t nrow,size_t max_iter,size_t verbose,size_t MAXITER,unsigned int seed);
size_t rewire_sparse_bipartite_ex(size_t *fro,size_t *to,size_t nc,size_t nr,size_t max_iter,size_t ne,size_t verbose,size_t MAXITER,unsigned int seed);
size_t analysis_undirected_ex(unsigned short *incidence,size_t ncol, size_t nrow,double *scores,size_t step,size_t max_iter,size_t verbose,size_t MAXITER,unsigned int seed);
size_t rewire_ex(unsigned short *matrix,size_t ncol, size_t nrow,size_t max_iter,size_t verbose,size_t MAXITER,unsigned int seed);
//size_t rewire_sparse_ex(size_t *fro,size_t *to,size_t *degree,size_t nc,size_t nr,size_t max_iter,size_t ne,size_t verbose,size_t MAXITER);
size_t analysis(unsigned short *incidence,size_t ncol, size_t nrow,double *scores,size_t step,size_t max_iter,size_t verbose,unsigned int seed);
size_t rewire_bipartite(unsigned short *matrix,size_t ncol, size_t nrow,size_t max_iter,size_t verbose,unsigned int seed);
size_t rewire_sparse_bipartite(size_t *fro,size_t *to,size_t nc,size_t nr,size_t max_iter,size_t ne,size_t verbose,unsigned int seed);
size_t analysis_undirected(unsigned short *incidence,size_t ncol, size_t nrow,double *scores,size_t step,size_t max_iter,size_t verbose,unsigned int seed);
size_t rewire(unsigned short *matrix,size_t ncol, size_t nrow,size_t max_iter,size_t verbose,unsigned int seed);
//size_t rewire_sparse(size_t *fro,size_t *to,size_t *degree,size_t nc,size_t nr,size_t max_iter,size_t ne,size_t verbose);


