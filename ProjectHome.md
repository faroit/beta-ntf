# Super β-NTF 2000 #
(by **Antoine Liutkus**)


---

In a nutshell, the beta\_ntf module
  * supports **weighted** parafac factorization of nonnegative numpy ndarray of **arbitrary shape** minimizing **any β-divergence**
  * fits in **one small single .py file**
  * requires numpy v1.6

---




## Overview ##

the **beta\_ntf** Python module provides a very simple to use, light and yet powerful implementation of the nonnegative PARAFAC model, also known as Nonnegative Tensor Factorization, Nonnegative Matrix Factorization (NMF) or Nonnegative Canonical Decomposition.


As an example, let X be a ndarray of shape (1000,400,5) composed of nonnegative entries:
```
>>>X.shape
(1000, 400, 5) 
```

The objective of beta\_ntf in that case is to approximate X using 3 matrices of respective shapes (1000,K) (400,K) and (5,K)  where K is called the number of components. Let us call these matrices A, B and C. We want to approximate X as:

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/mathml-gadget.xml" border="0" up\_content="X(a,b,c)=sum\_{k=1}^K A(a,k)B(b,k)C(c,k)" height="60" width="500" up\_fontsize="0.8em"/>

Such problems occur in many fields such as source separation, data-mining, image processing, etc. To perform this with say K=10 while minimizing the Kullback-Leibler divergence, simply use:

```
>>>import beta_ntf
>>>ntf = beta_ntf.BetaNTF(X.shape, n_components=10, beta=1, n_iter=100,verbose=True) 
>>>ntf.fit(X)
```

Now, the `ntf` object contains an approximation of X using 10 components.
You can
  * Directly access the factors as `ntf.factors_` which is a list composed of 3 matrices here.
  * Get the approximation using `beta_ntf.parafac(ntf.factors_)`
  * Get any of the constitutive components, e.g. using `ntf[...,3]` for the 4th component
  * Compute the total β-divergence of your approximation using `ntf.score(X)`


## Detailed Features ##
### 1) Simply compute the parafac model ###

Given a list `factors` of matrices which are ndarrays with the same number of columns, the beta\_ntf module permits to efficiently compute the corresponding PARAFAC model.
For example, assume
`>>>factors=[A,B,C,D]`
where

```
>>>A.shape
(100,5)
>>>B.shape
(50,5)
>>>C.shape
(10,5)
>>>D.shape
(5,5)
>>>factors=[A,B,C,D]
```

To compute the (100,50,10,5) tensor P given by

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/mathml-gadget.xml" border="0" up\_content="P(a,b,c,d)=sum\_{k=1}^5 A(a,k)B(b,k)C(c,k)D(d,k)" height="60" width="500" up\_fontsize="0.8em"/>

simply use
```
>>>P=beta_ntf.parafac(factors)
```

### 2) Easily generate random nonnegative data ###

As a small extra, the beta\_ntf module provides the nnrandn, which permits to generate nonnegative random ndarrays of arbitrary shape. Nothing complex really, but useful :
```
>>>E = beta_ntf.nnrandn((10,40,10))
>>>E.shape
(10,40,10)
```


### 3) Arbitrary shapes are supported ###

Interestingly, the beta\_ntf module supports data to fit of arbitrary shape (up to 25 dimensions). In the example above, X could well have been a (100,40,5,10,8) ndarray, with exactly the same syntax for using beta\_ntf. This feature is noticeable, since it is not so common (as of this writing in December 2012) to find light implementations of nonnegative PARAFAC models for higher order tensors, fitting in only one source file of less than 250 lines.

```
>>>import beta_ntf 
```

Generating synthetic data
```
>>>data_shape=(15,10,5,40)
>>>true_ncomponents=50
>>>true_factors = [beta_ntf.nnrandn((shape, n_components)) for shape in data_shape]
>>>X=beta_ntf.parafac(true_factors)
>>>X.shape
(15,10,5,40)
```
Creating the BetaNTF object
```
>>>ntf = beta_ntf.BetaNTF(X.shape, n_components=10, beta=1, n_iter=100,verbose=False)
```
Fitting the data
```
>>>beta_ntf.fit(X)
Fitting NTF model with 100 iterations....
Done.
<beta_ntf.BetaNTF instance at 0x7f664c01e0e0>
```
Now, construct the approximation
```
>>>model = beta_ntf.parafac(ntf.factors_)
>>>model.shape
(15, 10, 5, 40)
```
Print the shape of the factors
```
>>>for f in ntf.factors_:print f.shape 
(15, 10)
(10, 10)
(5, 10)
(40, 10)
```
Extract the 3 first components
```
>>>firstComponents=ntf[...,0:3]
>>>firstComponents.shape
(15, 10, 5, 40, 3)
```

### 4) All β-divergences are supported ###

While performing the approximation, beta\_ntf minimizes a cost function between original data X and its approximation P.
The cost function D to be minimized for this optimization is any β-divergence, summed over all entries of X. We thus have :

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/mathml-gadget.xml" border="0" up\_content="D(X|P)=sum\_i d\_beta (x(i)|P(i))" height="60" up\_fontsize="0.8em"/>


where the β-divergence d<sub>β</sub> to use for each entry is defined as :

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/mathml-gadget.xml" border="0" up\_content="d\_beta(x|y)={(x/y -log(x/y)-1,if beta=0),(xlog(x/y)+y-x,if beta=1),(1/(beta^2-beta)\*(x^beta +(beta-1)y^beta - beta x y^(beta-1)),text{otherwise}):}" height="200" width="600" up\_fontsize="0.8em"/>


Particular cases of interest are:
  * β = 2 : Euclidean distance
  * β = 1 : Kullback-Leibler divergence
  * β = 0 : Itakura-Saito divergence.

Simply change the `beta` property of a BetaNTF object to change the divergence it will minimize during fit.


### 5) Easily compute the score of your decomposition ###

It may be required to compute the score corresponding to a given decomposition, e.g. for model selection. If you want to know how far your NTF model is from data X, simply use :

```
>>> ntf.score(X)
304.01426732381253
```

The quantity returned gives the overall selected β-divergence between the model and the given ndarray, which ought to be of the same shape as ntf.data\_shape.
Since this score is a divergence, **the smaller, the better the approximation**.

## 6) Weighting the cost function ##

Some datapoints may be more important than others. Similarly, some datapoints may not be available, leading to missing data. This is common in data inpainting, or in source separation. In that case, it is desirable to weight the cost function accordingly.

More precisely, let X be the nonnegative ndarray to approximate and let P be the corresponding NTF model. Until now, I have only discribed estimation of P that minimizes the simple sum D of individual entrywise β-divergence :

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/mathml-gadget.xml" border="0" up\_content="D=sum\_{i}d\_beta( X[i] | P[i] )" height="60" up\_fontsize="0.8em"/>

where d<sub>β</sub> has been defined above. Instead of giving the same weight to all these individual divergences, we may instead _weight_ these entries so as to consider instead :

<wiki:gadget url="http://mathml-gadget.googlecode.com/svn/trunk/mathml-gadget.xml" border="0" up\_content="D\_W=sum\_{i}W[i]d\_beta( X[i] | P[i] )" height="60" up\_fontsize="0.8em"/>

where W gives the weights to assign to individual divergences. If a datapoint `X[i]` is less important than others, its corresponding weight `W[i]` may be smaller, or even 0 if `X[i]` is actually not known. This feature is implemented in beta\_ntf. If required, you can simply provide these weights as the `W` parameter to the `fit` function.

As a first exemple, suppose we have two **nonnegative** X and W such that
```
>>>X.shape
(400,50,30)
>>>W.shape
(400,50,30)
```

You can use W to weight the divergence simply through:
```
>>>ntf.fit(X,W)
```

where the BetaNTF object `ntf` has been defined as above.

Interestingly, W and X do not have to be of the same shape. It is only required that W\*X be defined, and thus that they can be broadcasted and multiplied. We could thus have instead :
```
>>>W.shape
(400,1,30)
```

and use the fit function the same way. This can be practical for e.g. giving different weights to whole modalities of the experiment.


# About the algorithm #

The algorithm implemented by beta\_ntf features standard multiplicative updates minimizing β-divergence, which were recently shown to guarantee a decrease of the cost-function at each step. For recent references on the topic, check for example:

  * A. Cichocki, R. Zdunek, A. H. Phan, and S. Amari, Nonnegative matrix and tensor factorizations : Applications to exploratory multi-way data analysis and blind source separation, Wiley Publishing, September 2009.
  * C. Févotte and J. Idier, Algorithms for nonnegative matrix factorization with the beta-divergence, Neural Computation 23 (2011), no. 9, 2421–2456.

Algorithms for the weighted NTF are derived exactly the same way. As far as I known they were first derived by Virtanen in his 2007 paper :

  * Virtanen, Tuomas, Monaural Sound Source Separation by Perceptually Weighted Non-Negative Matrix Factorization, Technical report, Tampere University of Technology, Institute of Signal Processing, 2007.

In any case, the update equations I derived in the general case of arbitrary dimension of the data is really nothing original with respect to those works and references therein. The main challenge was actually in terms of implementation. Still, I will write a small technical report about this and upload it here when available.

beta\_ntf makes use of the powerful voodoo magic of `numpy.einsum` to perform its computations, thus explaining both its computational efficiency and its conciseness. Hence, it requires numpy 1.6, in which `einsum` first appears.

Extensions of this small module for very large datasets would be nice. For this small project, I have focused on a light implementation that is easy to start with. It should run quite fast provided the data used fits in memory.