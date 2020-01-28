## This code is written by Andrea Gobbi, <gobbi.andrea@mail.com>.
## (C) 2015 BiRewire Developers.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

cimport cython
import numpy as np
cimport numpy as np
from math import *
import jgraph as i
import os
import csv

cdef extern from "lib/BiRewire.h":
    double similarity_undirected(unsigned short *m,unsigned short *n,size_t ncol,size_t nrow,size_t e)
    double similarity(unsigned short *m,unsigned short *n,size_t ncol,size_t nrow,size_t e )
    size_t analysis_ex(unsigned short *incidence,size_t ncol, size_t nrow,double *scores,size_t step,size_t max_iter,size_t verbose,size_t MAXITER,unsigned int seed)
    size_t analysis(unsigned short *incidence,size_t ncol, size_t nrow,double *scores,size_t step,size_t max_iter,size_t verbose,unsigned int seed)
    size_t rewire_bipartite(unsigned short *matrix,size_t ncol, size_t nrow,size_t max_iter,size_t verbose,unsigned int seed)
    size_t rewire_bipartite_ex(unsigned short *matrix,size_t ncol, size_t nrow,size_t max_iter,size_t verbose,size_t MAXITER,unsigned int seed)
    size_t rewire_sparse_bipartite_ex(size_t *fro,size_t *to,size_t nc,size_t nr,size_t max_iter,size_t ne,size_t verbose,size_t MAXITER,unsigned int seed)
    size_t rewire_sparse_bipartite(size_t *fro,size_t *to,size_t nc,size_t nr,size_t max_iter,size_t ne,size_t verbose,unsigned int seed)
    size_t analysis_undirected(unsigned short *incidence,size_t ncol, size_t nrow,double *scores,size_t step,size_t max_iter,size_t verbose,unsigned int seed)
    size_t analysis_undirected_ex(unsigned short *incidence,size_t ncol, size_t nrow,double *scores,size_t step,size_t max_iter,size_t verbose,size_t MAXITER,unsigned int seed)
    size_t rewire(unsigned short *matrix,size_t ncol, size_t nrow,size_t max_iter,size_t verbose,unsigned int seed)
    size_t rewire_ex(unsigned short *matrix,size_t ncol, size_t nrow,size_t max_iter,size_t verbose,size_t MAXITER,unsigned int seed)
    size_t rewire_sparse_ex(size_t *fro,size_t *to,size_t *degree,size_t nc,size_t nr,size_t max_iter,size_t ne,size_t verbose,size_t MAXITER,unsigned int seed)
    size_t rewire_sparse(size_t *fro,size_t *to,size_t *degree,size_t nc,size_t nr,size_t max_iter,size_t ne,size_t verbose,unsigned int seed)

def c_rewire_sparse_undirected(np.ndarray left,np.ndarray right,np.ndarray degree,N=-1, verbose=1,  MAXITER=10, accuracy=0.00005,exact=True,seed=0):

    cdef size_t e,nc,nr,t
    e= len(left)
    nc=nr= len(degree)
    t=nc*nr/2
    d=e/t
    if N==-1:
        if exact:
            N=ceil((e*(1-d)) *log((1-d)/accuracy) /2  )
        else:
            N=(e/(2*d^3-6*d^2+2*d+2))*log((1-d)/accuracy)  
    if exact:
        return N,<int>rewire_sparse_ex(<size_t*>np.PyArray_DATA(left),<size_t *>np.PyArray_DATA(right),<size_t *>np.PyArray_DATA(degree),nr, nc, N, e, verbose, N*MAXITER,seed)
    else:
        return N,<int>rewire_sparse(<size_t*>np.PyArray_DATA(left),<size_t *>np.PyArray_DATA(right),<size_t *>np.PyArray_DATA(degree),nr, nc, N, e, verbose,seed)

def c_rewire_sparse_bipartite(np.ndarray left,np.ndarray right,N=-1, verbose=1,  MAXITER=10, accuracy=0.00005,exact=True,seed=0):

    cdef size_t e,nc,nr,t
    e= len(left)
    nc,nr= len(np.unique(left)),len(np.unique(right))
    t=nc*nr
    if N==-1:
        if exact:
            N=ceil((e*(1-e/t)) *log((1-e/t)/accuracy) /2  )
        else:
            N=ceil((e/(2-2*e/t)) *log((1-e/t)/accuracy) )   
    if exact:
        return N,<int>rewire_sparse_bipartite_ex(<size_t*>np.PyArray_DATA(left),<size_t *>np.PyArray_DATA(right),nr, nc, N, e, verbose, N*MAXITER,seed)
    else:
        return N,<int>rewire_sparse_bipartite(<size_t*>np.PyArray_DATA(left),<size_t *>np.PyArray_DATA(right),nr, nc, N, e, verbose,seed)


def c_rewire_bipartite(np.ndarray incidence,N=-1, verbose=1,  MAXITER=10, accuracy=0.00005,exact=True,seed=0):
    cdef size_t e,nc,nr,t
    nc,nr= incidence.shape[1],incidence.shape[0]
    t=nc*nr
    e=incidence.sum()
    if N==-1:
        if exact:
            N=ceil((e*(1-e/t)) *log((1-e/t)/accuracy) /2  )
        else:
            N=ceil((e/(2-2*e/t)) *log((1-e/t)/accuracy) )  
    if exact:
        return   N,<int>rewire_bipartite_ex(<unsigned short*>np.PyArray_DATA(incidence), nc,  nr, N, verbose, N*MAXITER,seed)
    else:
         return  N,<int>rewire_bipartite(<unsigned short*>np.PyArray_DATA(incidence), nc,  nr, N, verbose,seed)

def c_rewire_undirected(np.ndarray incidence,N=-1, verbose=1,  MAXITER=10, accuracy=0.00005,exact=True,seed=0):
    cdef size_t e,nc,nr,t,d
    nc,nr= incidence.shape[1],incidence.shape[0]
    t=nc*nr/2
    e=incidence.sum()/2
    d=e/t
    if N==-1:
        if exact:
            N=ceil((e*(1-d)) *log((1-d)/accuracy) /2  )
        else:
            N=(e/(2*d^3-6*d^2+2*d+2))*log((1-d)/accuracy)
    if exact:
        return  N,<int>rewire_ex(<unsigned short*>np.PyArray_DATA(incidence), nc,  nr, N, verbose, N*MAXITER,seed)
    else:
         return N,<int>rewire(<unsigned short*>np.PyArray_DATA(incidence), nc,  nr, N, verbose,seed)

def c_analysis_bipartite(np.ndarray incidence,N=-1, verbose=1,  MAXITER=10, accuracy=0.00005,exact=True,step=10,seed=0):
    cdef size_t e,nc,nr,t,dim
    cdef np.ndarray scores
    nc,nr= incidence.shape[1],incidence.shape[0]
    t=nc*nr
    e=incidence.sum()
    if N==-1:
        if exact:
            N=ceil((e*(1-e/t)) *log((1-e/t)/accuracy) /2  )
        else:
            N=ceil((e/(2-2*e/t)) *log((1-e/t)/accuracy) )  
    score=np.ascontiguousarray(np.zeros(N+1),dtype=np.double)
    if exact:
        dim=  <int>analysis_ex(<unsigned short*>np.PyArray_DATA(incidence), nc,nr,<double*>np.PyArray_DATA(score), step, N, verbose, MAXITER*N,seed)
    else:
        dim=  <int>analysis(<unsigned short*>np.PyArray_DATA(incidence), nc,nr,<double*>np.PyArray_DATA(score), step, N, verbose,seed)

    return N,score[0:dim]

def c_analysis_undirected(np.ndarray incidence,N=-1, verbose=1,  MAXITER=10, accuracy=0.00005,exact=True,step=10,seed=0):
    cdef size_t e,nc,nr,t,dim,d
    cdef np.ndarray scores
    nc,nr= incidence.shape[1],incidence.shape[0]
    t=nc*nr/2
    e=incidence.sum()/2
    d=e/t
    if N==-1:
        if exact:
            N=ceil((e*(1-d)) *log((1-d)/accuracy) /2  )
        else:
            N=(e/(2*d^3-6*d^2+2*d+2))*log((1-d)/accuracy)
    score=np.ascontiguousarray(np.zeros(N+1),dtype=np.double)
    if exact:
        dim=  <int>analysis_undirected_ex(<unsigned short*>np.PyArray_DATA(incidence), nc,nr,<double*>np.PyArray_DATA(score), step, N, verbose, MAXITER*N,seed)
    else:
        dim=  <int>analysis_undirected(<unsigned short*>np.PyArray_DATA(incidence), nc,nr,<double*>np.PyArray_DATA(score), step, N, verbose,seed)

    return N,score[0:dim]

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([("", a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def incidence(np.ndarray x):
    edge_list=np.transpose((x==1).nonzero())
    edge_list[:,1]=edge_list[:,1]+x.shape[0]
    return i.Graph(list(edge_list))
#TODO
#class Dsg:
#    def __init__(self,data,type_of_array=None,type_of_graph=None):

    
class Rewiring:
    #data=None
    #N=None
    #data_rewired=None
    #jaccard_index=None
    #verbose=None
    #MAXITER=None
    #accuracy=None
    #step=None
    #exact=None
    #__type_of_data=None
    #__type_of_graph=None
    #__type_of_array=None
    def __init__(self,data,type_of_array=None,type_of_graph=None,verbose=False):
        """Rewiring object

        This class contains useful information when dealing with undirected or bipartite
        networks. During the initialization the class recorded the type of graph is used
        and also the kind of representaion is used. 
        :Parameters:
            data : the initial data. It can be a numpy matrix encoding an edgelist, an incidence
                matrix (for bipartite network) an adjacency matrix (for undirected network) or an 
                igraph object or a dsg class object. For a correct use, if an edgelist is given, 
                the values (nodes id) must start from 0 and have no holes.
            type_of_array: one among "edgelist_b","incidence", "adjacence","edgelist_u". If the 
                data filed is a numpy array we must specify what this array encodes. The first 
                two encode a bipartite network, the other two an undirected graph.
            type_of_graph: one among "bipartite", "undirected". Since the igraph function
                determinating if a graph is bipartite seems not to work properly, we must force 
                this information.
        With a Rewiring object we can generate a rewired wersion of it using the method rewire. 
        For more information about the rewiring algorithm see 
        Moreover we can compute the jaccard index between the initial data and its rewired version
        using the method jaccard_index, and performs an analysis of such index during the swithching 
        steps using the method descibed in Gobbi et al. Fast randomization of large
        genomic datasets while preserving alteration counts
        Bioinformatics 2014 30(17): i617-i623 doi: 10.1093/bioinformatics/btu474.
        """
        self.data=data
        if type(self.data)==i.Graph:
            self.__type_of_data="graph"
            self.__type_of_array="None"
            if self.data.is_directed():
                print "Directed graph are not supported.\n" 
                self.data=None
            else:    
                if type_of_graph in ["bipartite","undirected"] :  
                    self.__type_of_graph=type_of_graph
                else:
                    print "The type of graph must be bipartite or undirected.\n" 
                    self.data=None
            if self.data.has_multiple():
                print "Multiple edge are not supported, I will simplify the graph. \n" 
                self.data=self.data.simplify()
        else:    
            if type(self.data)==np.ndarray:
                self.__type_of_data="array"
                if type_of_array not in ["edgelist_b","incidence","adjacence","edgelist_u"]:
                    print "The input type of array is not supported or must be given. \n" 
                    self.data=None
                else:    
                    self.__type_of_array=type_of_array
                    if self.__type_of_array in ["edgelist_b","incidence"]:
                        self.__type_of_graph="bipartite"
                    else:
                        self.__type_of_graph="undirected"
                if self.__type_of_array=="edgelist_b" or self.__type_of_array=="edgelist_u" :
                    if self.data.shape[1]!=2:
                        print "Edgelist must contain 2 colums" 
                        self.data=None
                        self.__type_of_array=None
                    else:
                        self.data=unique_rows(self.data)
                        self.data=np.ascontiguousarray(self.data,dtype=np.uintp)

                else:
                    self.data=np.ascontiguousarray(self.data,dtype="H")
            else:
                print "Data type not supported.\n" 
                self.data=None
        if verbose:
            print("Object created: array="+self.__type_of_array+" data="+self.__type_of_data+" graph="+self.__type_of_graph)
    def rewire(self,N=-1,verbose=1,MAXITER=10, accuracy=0.00005,exact=True,seed=0):
        """ Rewiring routine

        It performs N switching steps of the graph encoded as a Rewiring
         object storing the result in the field data_rewired 
         (using the same format of the filed data).
        :Parameters:
            N : the number of swithching steps (SS) to perform. If -1 (default)
                the optimal bound is used. For reference see the documentation 
                of the class Rewiring. 
            verbose : 1 default. If 0 no message from C will displayed.    
            MAXITER : a multiplier of N in order to let the algorithm finish also
                in the case of inifinite loops
            accuracy : the distance, in terms of edge ratio, between the current distance 
                and the theoretical one from the fixed point.
            exact : True defautl. If False the routine counts also the unsucessfull
                switching step. A suitable N is computed in order to catch such faliures.
            seed : sees passed to C srand function. If 0 the seed is set to time(NULL).
        :Returns:   
            Boolean: if the switching algorithm has been sucessfully completed.
        """
        self.N=N
        self.verbose=verbose
        self.MAXITER=MAXITER
        self.accuracy=accuracy
        self.exact=exact
        self.seed=seed
        if N<=0 and N!=-1:
            print 'N must be positive or -1'
            return False 
        if self.__type_of_data=="graph":
            if self.__type_of_graph=="bipartite":
                ##get edgelist
                result=np.array(self.data.get_edgelist())
                left=np.ascontiguousarray(result[:,0],dtype=np.uintp)
                #right=np.ascontiguousarray(result[:,1]-min(result[:,1]),dtype=np.uintp)
                right=np.ascontiguousarray(result[:,1],dtype=np.uintp)
                tmp=c_rewire_sparse_bipartite(left,right,self.N, self.verbose,  self.MAXITER, self.accuracy,self.exact, self.seed)
                result=np.vstack((left,right)).T
                result=i.Graph(list(result))
            else:
                result=np.array(self.data.get_edgelist())
                result[result[:,0].argsort()]
                left=np.ascontiguousarray(result[:,0],dtype=np.uintp)
                right=np.ascontiguousarray(result[:,1],dtype=np.uintp)
                degree=np.ascontiguousarray(np.zeros(len(set(left).union(set(right)))),dtype=np.uintp)
                for j in range (0,len(right)):
                    degree[result[j,0]]+=1
                    degree[result[j,1]]+=1
                tmp=c_rewire_sparse_undirected(left,right,degree,self.N, self.verbose,  self.MAXITER, self.accuracy,self.exact, self.seed) 
                result=np.vstack((left,right)).T
                result=i.Graph(list(result))        
        if self.__type_of_data=="array":
            if self.__type_of_array=="edgelist_u":
                result=np.copy(self.data)
                result[result[:,0].argsort()]
                left=np.ascontiguousarray(result[:,0],dtype=np.uintp)
                right=np.ascontiguousarray(result[:,1],dtype=np.uintp)
                degree=np.ascontiguousarray(np.zeros(len(set(left).union(set(right)))),dtype=np.uintp)
                for j in range (0,len(right)):
                    degree[result[j,0]]+=1
                    degree[result[j,1]]+=1
                tmp=c_rewire_sparse_undirected(left,right,degree,self.N, self.verbose,  self.MAXITER, self.accuracy,self.exact, self.seed)
                result=np.vstack((left,right)).T
                for j in range(0,result.shape[0]):
                    if result[j,0]>result[j,1]:
                        t=result[j,0]
                        result[j,0]=result[j,1]
                        result[j,1]=t
            if self.__type_of_array=="edgelist_b":
                result=np.copy(self.data)
                left=np.ascontiguousarray(result[:,0],dtype=np.uintp)
                #right=np.ascontiguousarray(result[:,1]-min(result[:,1]),dtype=np.uintp)
                right=np.ascontiguousarray(result[:,1],dtype=np.uintp)
                tmp=c_rewire_sparse_bipartite(left,right,self.N, self.verbose,  self.MAXITER, self.accuracy,self.exact, self.seed)
                result=np.vstack((left,right)).T
            if self.__type_of_array=="incidence":
                result=np.ascontiguousarray(np.copy(self.data),dtype="H")
                tmp=c_rewire_bipartite(result,self.N, self.verbose,  self.MAXITER, self.accuracy,self.exact, self.seed)
              
            if self.__type_of_array=="adjacence":
                result=np.ascontiguousarray(np.copy(self.data),dtype="H")
                tmp=c_rewire_undirected(result,self.N, self.verbose,  self.MAXITER, self.accuracy,self.exact, self.seed)
        self.N=tmp[0]
        self.data_rewired=result
        if tmp[1]==0:
                return True
        else:
                return False
    def similarity(self):
        """Jaccard index

        It computes the jaccard index between the data filed and the data_rewired file if different from None.

        :Returns:   
            double precision: the computed jaccard index.
        """
        if self.data_rewired is None:
            print "First rewire the graph :-)."
            return -1
        else:
            if self.__type_of_array=="edgelist_b" or self.__type_of_array=="edgelist_u":
                m1=self.data.tolist()
                m2=self.data_rewired.tolist()
                m1=set(tuple(r) for r in m1)
                m2=set(tuple(r) for r in m2)
                j_i=len(m1.intersection(m2))
                return j_i/(len(self.data.tolist())-j_i)
            if self.__type_of_data=="graph":
                m1=(self.data).get_edgelist()
                m2=(self.data_rewired).get_edgelist()

                j_i=len(set(m1).intersection(set(m2)))
                return j_i/(len(list(m1))-j_i)
            if self.__type_of_array=="incidence" or self.__type_of_array=="adjacence":
                return   (self.data*self.data_rewired).sum()/((self.data+self.data_rewired).sum()-(self.data*self.data_rewired).sum())

    def analysis(self,n_networks=50,N=-1,verbose=1,MAXITER=10, accuracy=0.00005,exact=True,step=10,seed=0): 
        """ Analysis routine

        It computes the jaccard index between the initail grah and the current rewired version every step steps.
        The rewirin process is performed severatl times in order to extimate the mean value and CI of the JI.
        The result is stored as numpy array of 2 dimention in the field jaccard_index.
        :Parameters:
            N : the number of swithching steps (SS) to perform. If -1 (default)
                the optimal bound is used. For reference see the documentation 
                of the class Rewiring. 
            verbose : 1 default. If 0 no message from C will displayed.    
            MAXITER : a multiplier of N in order to let the algorithm finish also
                in the case of inifinite loops
            accuracy : the distance, in terms of edge ratio, between the current distance 
                and the theoretical one from the fixed point.
            exact : True defautl. If False the routine counts also the unsucessfull
                switching step. A suitable N is computed in order to catch such faliures.
            step : the number of SS between two measurement of the jaccard index.
            n_networks: the number of independent samples.
            seed : seed passed to srand function. If 0 the seed is set to time(NULL).
        :Returns:   
            Boolean: if the switching algorithm has been sucessfully completed.
        """
        self.verbose=verbose
        self.MAXITER=MAXITER
        self.accuracy=accuracy
        self.exact=exact 
        self.step=step
        self.N=N
        self.seed=seed
        result=np.ascontiguousarray(np.copy(self.data),dtype="H")
        RES=list()
        for j in range(0,n_networks):
            if self.__type_of_graph=="bipartite" and self.__type_of_array=="incidence":
                tmp=c_analysis_bipartite(result,self.N, self.verbose,  self.MAXITER, self.accuracy,self.exact,self.step, self.seed)        
            else:  
                if self.__type_of_graph=="undirected" and self.__type_of_array=="adjacence":
                    tmp=c_analysis_undirected(result,self.N, self.verbose,  self.MAXITER, self.accuracy,self.exact,self.step, self.seed)      
                else:
                    print "I accept only incidence and adjacency matrix"    
                    return False
            RES.append(tmp[1])
            result=np.ascontiguousarray(np.copy(self.data),dtype="H")
        self.data_rewired=None
        self.jaccard_index=RES
        self.N=tmp[0]
        return True         
    def sampler(self,path,K=2000,max=1000,N=-1,verbose=0,MAXITER=10, accuracy=0.00005,exact=True,seed=0):
        """ Null model sampler

        It creates K randomized graph starting from the initial graph writing the 
        adjacency or incidence matrix or the edgelist.
        :Parameters:
            N : the number of swithching steps (SS) to perform. If -1 (default)
                the optimal bound is used. For reference see the documentation 
                of the class Rewiring. 
            verbose : 1 default. If 0 no message from C will displayed.    
            MAXITER : a multiplier of N in order to let the algorithm finish also
                in the case of inifinite loops
            accuracy : the distance, in terms of edge ratio, between the current distance 
                and the theoretical one from the fixed point.
            exact : True defautl. If False the routine counts also the unsucessfull
                switching step. A suitable N is computed in order to catch such faliures.
            seed : seed passed to srand C function.  If 0 the seed is set to time(NULL).
            
        :Returns:   
            Boolean: if the sampler procedure has been sucessfully completed.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        num_sub=ceil(K/max)
        initial=self.data.copy()
        for i in range(0,num_sub):
            if not os.path.exists(path+"/"+str(i)):
                os.makedirs(path+"/"+str(i))
                if K-max*(i+1)<0  :
                    max=K-max*i
            for j in range(0,max):
                
                if self.rewire(N=N,verbose=verbose,MAXITER=MAXITER, accuracy=accuracy,exact=exact,seed=seed)==False:
                    self.data=initial.copy()
                    return False
                self.data=self.data_rewired
                out_file = path+"/"+str(i)+"/"+str(i)+"_"+str(j)
                if self.__type_of_data=="array":
                    np.save(arr=self.data_rewired,file=out_file)
                else:
                    np.save(arr=np.array((self.data_rewired).get_edgelist()),file=out_file)

            print "Saved "+str(max)+" files"
        self.data=initial
        return True
    def monitoring(self,n_networks=50,sequence=(1,10,100,-1),verbose=0,MAXITER=10, accuracy=0.00005,exact=True, seed=0): 
        """ Monitoring of the underlying markov chain

        It samples n_networks times the markov chain at the steps indicates in the list sequence.
        For each of this step the pairwise distance matrix is computed. This distance matrix can 
        be used to monitoring the makov chain for exmple using tsne dimentional scaling.
        :Parameters:
            n_networks : the number of samples for each step.
            sequence : the steps sequence to test.
            ...: other parameters passed to the function rewire.
            
        :Returns:   
            list: a list of distance matrix, one for each step to test.
        """
        tot=[]
        data_initial=self.data.copy()
        self.MAXITER=MAXITER
        self.verbose=verbose
        self.accuracy=accuracy
        self.exact=exact
        self.seed=seed
        for s in sequence:
            self.data=data_initial.copy()
            m=np.zeros((n_networks,n_networks))
            d=[data_initial]
            self.N=s
            for j in range(1,n_networks):
                self.rewire()
                d.append(self.data_rewired)
                self.data=self.data_rewired.copy()
                for k in range(0,j):
                    m[j,k]=m[k,j]=1-similarity(d[j],d[k],self.__type_of_array,self.__type_of_data)
            tot.append({"k":s,"dist":m})
        return tot
               
               


def read_BRCA(file):
    """ Load the BRCA dataset
    :Parameters:
        path : the path in which the BRCA file is stored.
        
    :Returns:   
        incidence matrix, colnames, rownames: the relative incidence matrix, the names of the 
        names of the rows.
    """
    try:
        cr = csv.reader(open(file))
    except IOError:
        print 'file not found.'
    lista=[]
    i=0
    rownames=[]
    for row in cr:
        if i !=0:
            lista.append(row[1:])
            rownames.append(row[0])
        else:
            colnames=row[1:]
        i=i+1    
    return np.array(lista),colnames,rownames

def similarity(m1,m2,array,data):
    if array=="edgelist_b" or array=="edgelist_u":
        m1=m1.tolist()
        m2=m2.tolist()
        e=len(self.data.tolist())
        m1=set(tuple(r) for r in m1)
        m2=set(tuple(r) for r in m2)
        j_i=len(m1.intersection(m2))
        return j_i/(e-j_i)
    if data=="graph":
        m1=m1.get_edgelist()
        m2=m2.get_edgelist()
        e=len(list(m1))
        j_i=len(set(m1).intersection(set(m2)))
        return j_i/(e-j_i)
    if array=="incidence" or array=="adjacence":
        return   (m1*m2).sum()/((m1+m2).sum()-(m1*m2).sum())
