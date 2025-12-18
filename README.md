Here there are some tools to treat evolution of solutions to equations having shocks.

Most of the tools are in shocks_utils.jl and in complementary_utils.jl

We have a test_weno.ipynb to test the weno implementations in shocks_utils.jl

We have a burgers_example.ipynb to run the standar example.

We have also a burgers_example_wgrad.ipynb to run a modified equation with a term like $\zeta (grad u)^2$ term in the flux.
Here we can use either the derivative reconstruction from the weno reconstruction polynomial (already present in the main 
weno reconstruction function) or a fourth order reconstruction from the weno reconstruction values at interfaces, obtained 
by applyting another function. 
