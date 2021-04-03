# Thesis
A Collection of Code I´ve written in my first large Coding Project, my Masters Thesis. 

To enable direct evaluation of progress, the work was done using Python and Jupyter Notebooks. In the first program, the code is run, the second program is to prepare data for plotting and make it more human understandable, while the third is for plotting the data. Another version of the plotting program is also present, which produces special other plots for the thesis. The .stan files are files for the programming language stan (mc-stan.org), which enables easy and efficient scientific stochastic computing by compiling a c++ program out of the mathematical model defined in the .stan files. 

The largest part of the thesis was to develop the models, and in the end compare them. Since some of the models are tensor-like, the models had to be optimized to be able to run at all on the cluster, due to computing resources being used up like O(N^3) (at least almost, some optimization was able to bringt that down towards something like 2.7).
