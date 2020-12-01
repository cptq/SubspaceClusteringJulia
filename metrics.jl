using Clustering
using LinearAlgebra
using Arpack # sparse eigendecomp
using SparseArrays
import Hungarian # for linear assignment
using Statistics: median


function _spectral_embedding(A::SparseMatrixCSC, k::Int)
	D = Diagonal(vec(sum(A, dims=2)) .+ 1e-8) # small constant avoids divide by zero
	D_norm = sqrt(inv(D))
	L_sym = Symmetric(I - D_norm * A * D_norm)
	# compute eigenvals near zero
	Λ, V = eigs(L_sym, nev=k, sigma=1e-6, which=:LM) 
	V = transpose(V)
	return Λ, V
end

function _spectral_embedding(A::AbstractMatrix, k::Int)
	D = Diagonal(vec(sum(A, dims=2)) .+ 1e-8) # small constant avoids divide by zero
	D_norm = sqrt(inv(D))
	L_sym = Symmetric(I - D_norm * A * D_norm)
	Λ, V = eigen(L_sym, 1:k)
	V = transpose(V)
	return Λ, V
end

function spectral_clustering_metric(C::AbstractMatrix, k::Int, label, metrics; niters::Int=100, affinity::AbstractString="sym", verbose::Bool=true, extra_dim::Bool=false)
	""" Runs spectral clustering then computes a clustering metric
	label is true labels
	niters is number of kmeans clusterings to run and average over
	"""
	A = abs.(copy(C))
	if affinity == "sym"
		A = (A + A')/2
	end

	_, V = _spectral_embedding(A, extra_dim ? k+1 : k)
	V = mapslices(normalize, V, dims=1)
	
	scores = Dict{String, Array{}}(metric => [] for metric in metrics)

	for i = 1:niters
		result = kmeans(V, k)
		pred_label = assignments(result)

		for metric in metrics
			if metric == "nmi"
				score = mutualinfo(label, pred_label)
			elseif metric == "accuracy"
				score = clustering_acc(label, pred_label)
			end
			push!(scores[metric], score)
		end
	end

	avgs = [sum(scores[metric])/niters for metric in metrics]
	meds = [median(scores[metric]) for metric in metrics]
	if verbose 
		println("Mean | Median")
		for (i, metric) in enumerate(metrics)
			println(metric, ": ",  avgs[i], " | ", meds[i])
		end
	end
	return avgs, meds
end

function feature_detection(C::AbstractMatrix, label::Array{Int})
	""" Computes the mass in correct subspace vs total mass in each column
	1/N * ∑_{i=1}^N (1 - ||c_{i, k_i}||_1 / ||c_i||_1 ) 
	label[i] is the subspace that point x_i is in
	"""
	N = size(C, 1)

	# mapping from cluster to points of cluster
    s_points = Dict{Int, Array{Int}}()
    for (i, s) in enumerate(label)
        if haskey(s_points, s)
            push!(s_points[s], i)
        else
            s_points[s] = [i]
        end
    end
    tot_err = 0
    for i in 1:N
        s = label[i]
        in_mass = sum(abs.(C[s_points[s], i]))
        tot_mass = sum(abs.(C[:,i]))
		if tot_mass == 0
			tot_err += 1
		else
        	tot_err += 1 - in_mass/tot_mass
		end
    end
    return tot_err / N
end

function clustering_acc(label::Array{Int}, pred_label)
	cont_table = counts(label, pred_label)
	matching = Hungarian.munkres(-cont_table')'
	clust_assignments = [findfirst(matching[i,:] .== Hungarian.STAR) for i = 1:size(cont_table,1)]
	wrong = 0
	for i in 1:length(label)
		wrong += (clust_assignments[label[i]] != pred_label[i])
	end
	return 1 - wrong/length(label)
end

function sparsity(C::AbstractMatrix; ϵ::Number=1e-9)
	""" Average number of nonzeros per row of C """
	return sum(abs.(C) .> ϵ) / size(C , 1)
end

