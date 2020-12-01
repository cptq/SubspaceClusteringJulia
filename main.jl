# USAGE: julia main.jl $DATASET $METHOD
include("data_gen.jl")
include("rep_solver.jl")
include("metrics.jl")

# DATASET CHOICES:
# umist, scattered_coil, scattered_umist, scattered_coil_40
data_name = ARGS[1] 

# SOLVER CHOICES:
# adssc, jdssc, jdssc_l1, tsc, lsr, lrsc, ssc_omp, ssc, ensc
solver_type = ARGS[2]

verbose = true
small_data_loader = () -> gen_synthetic(10, 2, 3, 50, 0.0) # for precompilation

# Choice of data loader
if data_name == "umist"
	data_loader = () -> load_umist_resized()
elseif data_name == "scattered_coil"
	data_loader = () -> load_scattered_coil(nspaces=100)
elseif data_name == "scattered_coil_40"
	data_loader = () -> load_scattered_coil(nspaces=40)
elseif data_name == "scattered_umist"
	data_loader = () -> load_scattered_umist()
else
	error("invalid data loader")
end

io = stdout
if solver_type in ["ssc", "ensc"]
	io = open("results.txt", "w") # use this to print to file e.g. to look at ssc output without warnings
end


# Solvers and parameter choices
solvers = []
println(io, "Solver type: ", solver_type)
println(io, "Dataset: ", data_name)

if solver_type == "adssc"
	for η1 in [.1, 1, 10, 25, 50]
		for η2 in [.0005, .001, .025, .05, .01, .1]
			solver = function(norm_X)
						println(io, "η1: $(η1) | η2: $(η2)")
						return adssc(norm_X, η1, η2; η3=0)		|>
								 C -> (abs.(C) + abs.(C'))/2 
				     end
			push!(solvers, solver)
		end
	end
elseif solver_type == "jdssc"
	niters = 2000
	ρ = .5
	for η1 in [.01, .25, 1.0, 25]
		for η2 in [.01, .05, .1, .2]
			solver = function(norm_X)
						println(io, "η1: $(η1) | η2: $(η2) | niters: $(niters) | ρ: $(ρ)")
						return jdssc(norm_X, η1, η2; niters=niters,ρ=ρ,verbose=false, τ1=0.0001)[1]
					 end
			push!(solvers, solver)
		end
	end
elseif solver_type == "jdssc_l1"
	niters = 2000
	ρ = .5
	for η1 in [.01, .25, 1.0, 25]
		for η2 in [.01, .05, .1, .2]
			for η3 in [0, .1]
				solver = function(norm_X)
					println(io, "η1: $(η1) | η2: $(η2) | η3: $(η3) | niters: $(niters) | ρ: $(ρ)")
							return jdssc(norm_X, η1, η2; niters=niters,ρ=ρ,verbose=false, τ1=0.0001, η3=η3)[1]
						 end
				push!(solvers, solver)
			end
		end
	end
elseif solver_type =="tsc"
	for q in 2:15
		solver = function(norm_X)
			println(io, "q: $(q)")
			return tsc(norm_X, q)
		end
		push!(solvers, solver)
	end
elseif solver_type =="lsr"
	for γ in [.01, .1, .5, 1, 10, 50, 100]
		solver = function(norm_X)
					println(io, "γ: $(γ)")
					return lsr(norm_X, γ, zero_diag=true)
				 end
		push!(solvers, solver)
	end
elseif solver_type =="lrsc"
	# using default γ = τ/2
	for τ in [.1, 1, 10, 50, 100, 500, 1000]
		solver = function(norm_X)
			println(io, "τ: $(τ)")
			return lrsc(norm_X, τ)
		end
		push!(solvers, solver)
	end
elseif solver_type =="ssc_omp"
	for q in 2:15
		solver = function(norm_X)
			println(io, "q: $(q)")
			return ssc_omp(norm_X, q; ϵ=1e-6)
		end
		push!(solvers, solver)
	end
elseif solver_type == "ssc"
	for γ in [1, 5, 10, 25, 50, 100, 1000]
		solver = function(norm_X)
				println(io, "γ: $(γ)")
				return ensc(norm_X, γ, 1; algorithm="spams")
		end
		push!(solvers, solver)
	end
elseif solver_type == "ensc"
	for γ in [.1,  1, 5, 10, 50, 100, 200]
		solver = function(norm_X)
				println(io, "γ: $(γ)")
				return ensc(norm_X, γ, .5; algorithm="spams")
		end
		push!(solvers, solver)
	end
else
	error("invalid solver")
end

# Precompile solver
println(io, "Precompiling solver on small dataset...")
X, label = small_data_loader()
solvers[1](X);


println(io, "Loading data...")
X, nspaces, label = data_loader()
println(io, "Data shape: ", size(X), " | Number of clusters: ", nspaces)

println(io, "Running solver...")
for curr_solver in solvers
	println(io, "")
	@time C = curr_solver(X)

	fd = feature_detection(C, label)
	sp = sparsity(C)

	if verbose
		println(io, "Sum of C entries: ", sum(C), " | maximum: ", maximum(C))
		println(io, "Mass in wrong subspace: ", fd)
		println(io, "Sparsity: ", sp)
	end

	redirect_stdout(io) do
		nmi, acc = spectral_clustering_metric(C, nspaces, label, ["nmi", "accuracy"],
								   niters=100, verbose=verbose);
	end
	
end

