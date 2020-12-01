using LinearAlgebra
using PyCall
using SparseArrays
using StatsBase

function correlation_kernel(X::AbstractMatrix)
	"""Computes ( |X^T X| - diag(|X^T X|) )"""
	A = abs.(X'*X)
	A .= A - Diagonal(A)
	return A
end

function correlation_kernel_sparse(X::AbstractMatrix; gamma::Number=1, nn::Int=100, chunk_size::Int=500)
	""" 
	nn is number of nearest neighbors per column
	chunk_size is number of columns to process at a time
	"""
	n = size(X, 2)
	A = spzeros(n,n)
	chunk_size = minimum((n, chunk_size))
	cols = zeros(n, chunk_size)
	nn = minimum((n-1, nn)) # cannot have less neighbors than points
	num_chunks = div(n, chunk_size)
	for chunk in 1:num_chunks
		chunk_inds = 1+(chunk-1)*chunk_size : chunk*chunk_size
		@views cols .= abs.(X'*X[:,chunk_inds])
		for (idx, i) in enumerate(chunk_inds)
			cols[i,idx] = 0
		end
		Inds = mapslices(c->partialsortperm(c, 1:nn, rev=true), cols, dims=1)
		for (idx, i) in enumerate(chunk_inds)
			@views inds = Inds[:, idx]
			A[inds,i] .= cols[inds,idx]
		end
	end

	# finish remaining entries not in chunk
	last_inds = num_chunks * chunk_size + 1 : n
	if length(last_inds) > 0
		@views last_cols = abs.(X'*X[:, last_inds])
		for (idx, i) in enumerate(last_inds)
			last_cols[i, idx] = 0
		end
		Inds = mapslices(c->partialsortperm(c, 1:nn, rev=true), last_cols, dims=1)
		for (idx, i) in enumerate(last_inds)
			@views inds = Inds[:, idx]
			A[inds,i] .= last_cols[inds,idx]
		end
	end

	A .= A - Diagonal(A)
	A ./= gamma
	return A
end


function scipyCSC_to_julia(A)
	""" convert scipy sparse format to julia sparse matrix 
	https://discourse.julialang.org/t/performance-bug-when-creating-a-sparse-matrix-via-python/33661/4
	"""
	m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    B = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    return B
end
