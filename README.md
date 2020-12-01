## Subspace Clustering Algorithms

This repo includes implementations of various subspace clustering algorithms in Julia, tested on Julia 1.5. To run, they require several packages, which can be installed by `julia requirements.jl`.

The main implementations for computing the self-representation matrices are in `rep_solver.jl`. We also call the Python implementations by [Chong You](https://github.com/ChongYou/subspace-clustering) for EnSC and SSC as located in `selfrepresentation.py`. The file `main.jl` gives a framework to run these methods with some grid searches over parameters. The usage is:
```
julia main.jl $DATASET $METHOD
```
where the DATASET options are: "umist", "scattered\_coil", "scattered\_umist", and "scattered\_coil\_40", and the METHOD options are: "adssc", "jdssc", "jdssc\_l1", "tsc", "lsr", 'lrsc", "ssc\_omp", "ssc", and "ensc".
For instance, to run LSR on UMIST, do:
```
julia main.jl umist lsr
```
Data loaders are in `data_gen.jl`.  Other datasets can be experimented on by making another function in the format as in the file, i.e., a function that returns the data matrix `norm_X` with points normalized to unit l2 norm, the integer `nspaces` with the number of (subspace) clusters, and array `label` where `label[i]` is the subspace cluster of point `i`.
