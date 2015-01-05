This text originally appears as an appendix to my honours paper: “DNN
Low Level Reinitialization: A Method for Enhancing Learning in Deep
Neural Networks through Knowledge Transfer” (Lyndon White, October
2014).

Large portions of the text originally appears in the Project Proposal
for “Domain Adaptation in Deep Belief Networks” (Lyndon White, May
2014).

The YADLF Framework
===================

A framework for neural network based machine learning has been created.
The library created is titled, semi-ironically, YADLF or Yet Another
Deep Learning Framework. At the conclusion of this project YADLF will be
released as a open source project to allow it to be used by others in
their research, or as a learning resource for future implementers.

Existing frameworks such as GPUMLib and Pylearn2 were examined and
considered unsuitable. In particular, it was found that frameworks were
typically purpose-constructed by the research authors. Existing
frameworks being too inflexible to allow new techniques to be easily
explored.

Further, as marginalizing stacked denoising autoencoders(Chen et al.
2014) (mSDA) are a fairly new they have not been incorporated into any
existing larger deep learning library.

Constructing this framework has been a positive educational exercise,
and has lead to a better understanding of the algorithms involved. With
improved understanding of how it all works together, it was easier to
implement adjustments required which has allowed for this project to
accomplish its goals. A variety of technologies have been used to create
YADLF.

The framework was created in Python 2.7(Van Rossum 2010) using a number
of tools, as were found to be common in research papers. Preeminently
amongst the tools used are numpy(Oliphant 2007)(Jones et al. 2001),
Cython(Behnel et al. 2011) and IPython(Perez and Granger 2007). Numpy is
a library for working with the SIMD (Single Instruction Multiple Data)
components of the CPU(Oliphant 2007)(Jones et al. 2001). Using numpy is
very similar in techniques and speed to using Matlab. Cython is a
extension to the Python language that compiles Python via C into native
machine code(Behnel et al. 2011). Through profiling the most critical
sections where Cython gives speed improvement have been determined, and
rewritten using Cython. Using these tools a well featured and flexible
machine learning library was implemented.

Features
--------

[par:Features]

The framework has the following features.

-   Dataset techniques:

    -   Standardized(LeCun et al. 2012)[^1]

    -   Balancing of order of training case classes(LeCun et al.
        2012)(Hinton 2010), while preserving separation of test and
        training data.

    -   Serialization for fast loading(Bilina and Lawford 2012)

-   RBMs:

    -   Bernoulli-Bernoulli RBMs (BB-RBM)(Bengio 2009a)

    -   Gaussian-Bernoulli RBMs (GB-RBM)(Bengio et al. 2007) – with
        standardized input only.

    -   Contrastive Divergence(Hinton 2002) training algorithm, with
        momentum(Hinton 2010), mini-batch learning(Hinton 2010) and L2
        Weight Decay (Hinton 2010)

-   Feed Forward Neural Networks:

    -   Back-propagation(Rumelhart, Hinton, and Williams 1986) with
        momentum(Rumelhart, Hinton, and Williams 1986), mini-batch
        learning(LeCun et al. 2012) and L2 Weight Decay (Hinton and Van
        Camp 1993)

    -   Early Stopping(Prechelt 1998), with optional learning rate
        adaption

    -   Variety of neuron activation functions including softmax(Bridle
        1990) and sigmoid(Rumelhart, Hinton, and Williams 1986).

-   DBNs

    -   Greedy Layer-wise Training (Bengio 2009b)

    -   Conversion to initialized DNN(Hinton 2009)(Bengio et al. 2007)

-   mSDAs

    -   mSDA full batch training(Chen et al. 2014)

    -   Conversion to initialized DNN(Hinton 2009)(Bengio et al. 2007)

-   Knowledge Matrix Restructuring Techniques

    -   As the focus of this project was investigation into transfer of
        knowledge YADLF, unlike any existent deep learning library,
        comes with substantial functionality for manipulating the
        knowledge of neural architectures. Those being the bias and
        weight matrices. It does this without destroying any knowledge
        contained within the layers.

    -   Reinitialization of layers (Used in DNN LLR to reset the bottom
        layer)

    -   Widening of layers, making them wider with new neurons.

    -   Adding layers to the top of the network

YADLF works well with the Data Analysis tools discussed in .

Data Analysis and Presentation Tools[sec:Results-Analysis-Tools]
================================================================

Analysis of results and production of plots for this report was carried
out using well developed scientific Python ecosystem.

-   IPython is a Python interface, which resembles the Mathematica
    notebook(Perez and Granger 2007).

    -   It was used as an interface for the majority of interaction with
        the data analysis tools.

    -   It also supports process dispatching across multiple cores. This
        was used in preliminary analysis

-   Pandas (McKinney 2010) what used for data analysis

    -   Tasks such as finding the means and standard deviations, on a
        basis of per topology and training set size, are findable in a
        few lines of code.

-   Matplotlib(Hunter 2007)

    -   Matplotlib is the standard underlying plotting and graphing
        framework for python.

    -   It is used by Pandas and Seaborn.

-   Seaborn(Waskom 2012)

    -   Seaborn is a data visualization tool

    -   It draws on the capacities of matplotlib and Pandas to allow
        many more types of plots and charts to be created

-   The TeX package Tikz(Tantau) was used to generate the diagrams used
    thought this work.

-   This paper was was created in (Ettrich and others 1995), and typeset
    using LuaTeX(Hagen 2005).

Behnel, S., R. Bradshaw, C. Citro, L. Dalcin, D. S. Seljebotn, and K.
Smith. 2011. “Cython: The Best of Both Worlds.” *Computing in Science
Engineering* 13 (march april): 31–39. doi:10.1109/MCSE.2010.118.

Bengio, Yoshua. 2009a. *Learning deep architectures for AI*. Vol. 2. Now
Publishers Inc.

———. 2009b. *Learning deep architectures for AI*. Vol. 2. Now Publishers
Inc.

Bengio, Yoshua, Pascal Lamblin, Dan Popovici, and Hugov Larochelle.
2007. “Greedy layer-wise training of deep networks.” *Advances in neural
information processing systems* 19: 153.
[http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf](http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf "http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks.pdf").

Bilina, Roseline, and Steve Lawford. 2012. “Python for Unified Research
in Econometrics and Statistics.” *Econometric Reviews* 31: 558–591.

Bridle, John S. 1990. “Training Stochastic Model Recognition Algorithms
as Networks can Lead to Maximum Mutual Information Estimation of
Parameters.” In *Advances in Neural Information Processing Systems 2*,
ed. D. S. Touretzky, 211–217. Morgan-Kaufmann.
[http://papers.nips.cc/paper/195-training-stochastic-model-recognition-algorithms-as-networks-can-lead-to-maximum-mutual-information-estimation-of-parameters.pdf](http://papers.nips.cc/paper/195-training-stochastic-model-recognition-algorithms-as-networks-can-lead-to-maximum-mutual-information-estimation-of-parameters.pdf "http://papers.nips.cc/paper/195-training-stochastic-model-recognition-algorithms-as-networks-can-lead-to-maximum-mutual-information-estimation-of-parameters.pdf").

Chen, Minmin, Kilian Weinberger, Fei Sha, and Yoshua Bengio. 2014.
“Marginalized Denoising Auto-encoders for Nonlinear Representations.” In
*Proceedings of The 31st International Conference on Machine Learning*,
1476–1484.
[http://jmlr.org/proceedings/papers/v32/cheng14.pdf](http://jmlr.org/proceedings/papers/v32/cheng14.pdf "http://jmlr.org/proceedings/papers/v32/cheng14.pdf").

Ettrich, Matthias, and others. 1995. “The LyX document processor.”

Hagen, Hans. 2005. “LuaTEX: Howling to the moon.” *Communications of the
Tex Users Group Tugboat*: 152.

Hinton, Geoffrey. 2010. “A practical guide to training restricted
Boltzmann machines.” *Momentum* 9.
[http://www.cs.toronto.edu/\~hinton/absps/guideTR.pdf](http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf "http://www.cs.toronto.edu/~hinton/absps/guideTR.pdf").

Hinton, Geoffrey E. 2002. “Training products of experts by minimizing
contrastive divergence.” *Neural computation* 14: 1771–1800.

Hinton, Geoffrey E., and Drew Van Camp. 1993. “Keeping the neural
networks simple by minimizing the description length of the weights.” In
*Proceedings of the sixth annual conference on Computational learning
theory*, 5–13.

Hinton, Goeff. 2009. “Recent Developments in Deep Learning.” The
University of British Columbia.
[http://www.youtube.com/watch?v=vShMxxqtDDs](http://www.youtube.com/watch?v=vShMxxqtDDs "http://www.youtube.com/watch?v=vShMxxqtDDs").

Hunter, J. D. 2007. “Matplotlib: A 2D graphics environment.” *Computing
In Science & Engineering* 9: 90–95.

Jones, Eric, Travis Oliphant, Pearu Peterson, and others. 2001. “SciPy:
Open source scientific tools for Python.”
[http://www.scipy.org/](http://www.scipy.org/ "http://www.scipy.org/").

LeCun, Yann A., Léon Bottou, Genevieve B. Orr, and Klaus-Robert Müller.
2012. “Efficient backprop.” In *Neural networks: Tricks of the trade*,
9–48. Springer.

McKinney, Wes. 2010. “” In , ed. Stéfan van der Walt and Jarrod Millman,
51–56.

Oliphant, Travis E. 2007. “Python for Scientific Computing.” *Computing
in Science & Engineering* 9: 10–20.
doi:http://dx.doi.org/10.1109/MCSE.2007.58.
[http://scitation.aip.org/content/aip/journal/cise/9/3/10.1109/MCSE.2007.58](http://scitation.aip.org/content/aip/journal/cise/9/3/10.1109/MCSE.2007.58 "http://scitation.aip.org/content/aip/journal/cise/9/3/10.1109/MCSE.2007.58").

Pedregosa, F., G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O.
Grisel, M. Blondel, et al. 2011. “Scikit-learn: Machine Learning in
Python.” *Journal of Machine Learning Research* 12: 2825–2830.

Perez, Fernando, and Brian E. Granger. 2007. “IPython: A System for
Interactive Scientific Computing.” *Computing in Science & Engineering*
9: 21–29. doi:http://dx.doi.org/10.1109/MCSE.2007.53.
[http://scitation.aip.org/content/aip/journal/cise/9/3/10.1109/MCSE.2007.53](http://scitation.aip.org/content/aip/journal/cise/9/3/10.1109/MCSE.2007.53 "http://scitation.aip.org/content/aip/journal/cise/9/3/10.1109/MCSE.2007.53").

Prechelt, Lutz. 1998. “Early stopping-but when?.” In *Neural Networks:
Tricks of the trade*, 55–69. Springer.

Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. 1986.
“Learning representations by back-propagating errors.” *Nature* 323:
533–536.

Tantau, Till. “The TikZ and PGF Packages Manual for version 3.0.0.”
[http://sourceforge.net/projects/pgf/](http://sourceforge.net/projects/pgf/ "http://sourceforge.net/projects/pgf/").

Van Rossum, Guido. 2010. “Python 2.7 Documentation.”

Waskom, Michael. 2012. “Seaborn 0.4.0.”
[http://web.stanford.edu/\~mwaskom/software/seaborn/](http://web.stanford.edu/~mwaskom/software/seaborn/ "http://web.stanford.edu/~mwaskom/software/seaborn/").

[^1]: Implemented in current version using scikit-learn preprocessing
    library(Pedregosa et al. 2011), for speed.
