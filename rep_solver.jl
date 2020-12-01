using LinearAlgebra
using Optim
using SparseArrays
using PyCall
using Suppressor

include("utils.jl")

function tsc(X::AbstractMatrix, q::Int; chunk_size::Int=500)
	"""
	'Robust Subspace Clustering via Thresholding'
	"""
	Z = correlation_kernel_sparse(X; nn=q, chunk_size=chunk_size)
	nonzeros(Z) .= exp.(-2*acos.(clamp.(nonzeros(Z), 0, 1)))
	return (Z + Z')/2
end

function lsr(X::AbstractMatrix, γ::Number; zero_diag::Bool=true)
	"""
	'Robust and Efficient Subspace Segmentation via Least Squares Regression'
	Zero diagonal variant.
	"""
	XtX = X'*X
	XtX_γ = XtX + γ * I
	if zero_diag
		D = inv(XtX_γ)
		C = -D * inv(Diagonal(D))
		C .= C - Diagonal(C)
	else
		C = XtX_γ \ XtX
	end
	return C
end

function ssc_omp(X::AbstractMatrix, kmax::Int; ϵ=1e-7)
	"""
	'Greedy Feature Selection for Subspace Clustering'
	'Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit'
	"""
	d, n = size(X)
	C = spzeros(n,n)

	function omp_col!(l)
		k = 0
		T = []
		q = X[:,l]
		c = zeros(1)
		while k < kmax && norm(q)^2 > ϵ
			coherence = abs.(X'*q)
			coherence[l] = 0 # do not choose self
			i = argmax(coherence)
			push!(T, i)
			c = X[:, T] \ X[:,l]
			q .= X[:,l] - X[:,T]*c
			k += 1
		end
		C[T,l] .= c
	end

	for l = 1:n
		omp_col!(l)
	end
	return C
end

function lrsc(X::AbstractMatrix, τ::Number=-1, γ::Number=-1)
	"""
	'A Closed Form Solution to Robust Subspace Estimation and Clustering'
	Noisy problem implementation
	"""
	# default parameters from given code
	if τ < 0; τ = 100 / opnorm(X)^2; end
	if γ < 0; γ = τ/2; end

	d, n = size(X)
	F = svd(X)
	if 3*τ < γ
		sigma = (γ + τ)/γ/sqrt(τ)
	else
		sigma = sqrt((γ+τ)/γ^2/τ)
		sigma = sqrt((γ+τ)/γ/τ + sqrt(sigma))
	end
	λ = F.S .* (F.S .> sigma) + γ/(γ+τ) * F.S .* (F.S .<= sigma)

	r = Int(maximum((sum(λ .> 1/sqrt(τ)), 1)))
	C = F.V[:,1:r] * (I - 1/τ * Diagonal(1 ./(λ[1:r] .^ 2))) * F.Vt[1:r,:]
	return C
end

function ensc(X::AbstractMatrix, γ::Number, τ::Number=1; algorithm::AbstractString="spams")
	""" Elastic Net Subspace Clustering
	'Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering'
	"""

	pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__) # include current directory
	sr = pyimport("selfrepresentation")
	model = sr.ElasticNetSubspaceClustering(active_support=true, tau=τ, gamma=γ, algorithm=algorithm, gamma_nz=false)
	model.fit_self_representation(transpose(X))
	C = model.representation_matrix_
	return scipyCSC_to_julia(C)
end

function jdssc(X::AbstractMatrix, η1::Number, η2::Number; niters::Int=2000, ρ::Number=.5, ϵ::Number=1e-8, verbose::Bool=true, τ1::Number=0.0001, η3::Number=0.0)
	""" J-DSSC model from 'Doubly Stochastic Subspace Clustering'
	"""
	γ1 = sqrt(η1)
	γ2 = sqrt(η1)*η2
	γ3 = η3
	return joint_lsr_linearized_alt(X, γ1, γ2; niters=niters, ρ=ρ, ϵ=ϵ, verbose=verbose, τ1=τ1, γ3=γ3)
end

function ot_q_dual(K::Symmetric, γ::Number; verbose::Bool=true, solver::AbstractString="LBFGS")
	""" Quadratically regularized optimal transport for symmetric input.
		Solves via dual.
	"""
	n = size(K, 2)
	α0 = -ones(n) .* maximum(K)/2 # initial start
	
	# preallocate space
	col_f = zeros(n)
	col_g = zeros(n)

	# objective function
	function _f(α)
		val = 0
		@inbounds for i in 1:n
			@views col_f .= α[i] .+ α .+ K[:,i]
			clamp!(col_f, 0, Inf)
			col_f .^= 2
			val += sum(col_f)
		end
		val /= 2*γ
		val -= 2*sum(α)
		return val
	end

	# gradient function
	function _g!(G, α)
		@inbounds for i in 1:n
			@views col_g .= α[i] .+ α .+ K[:,i]
			clamp!(col_g, 0, Inf)
			G[i] = sum(col_g)
		end
		G .*= 2/γ
		G .-= 2
	end

	f = _f
	g! = _g!

	if solver == "LBFGS"
		results = optimize(f, g!, α0, LBFGS())
	elseif solver == "GradientDescent"
		results = optimize(f, g!, α0, GradientDescent())
	else
		throw("Invalid solver")
	end
	α = Optim.minimizer(results)

	if verbose; println("MinMax α: ", minimum(α), " | ", maximum(α)); end

	C = spzeros(n,n)
	for i in 1:n
		@views C[:,i] .= clamp.(α[i] .+ α .+ K[:,i], 0, Inf)
	end
	C ./= γ

	return C
end

function ot_q_dual(K::Union{AbstractMatrix, SparseMatrixCSC}, γ::Number; verbose::Bool=true, solver::AbstractString="LBFGS")
	""" Quadratically regularized optimal transport
	"""
	n = size(K, 2)
	ν0 = -ones(2*n) .* maximum(K)/2 # initial start
	
	# preallocate space
	col_f = zeros(n)
	col_g = zeros(2*n)

	# objective function
	function f(ν)
		val = 0
		for k in 1:n
			@views col_f .= ν[k+n] .+ ν[1:n] .+ K[:,k]
			clamp!(col_f, 0, Inf)
			col_f .^= 2
			val += sum(col_f)
		end
		val /= 2*γ
		val -= sum(ν)
		return val
	end

	# gradient function
	function g!(G, ν)
		for k in 1:n
			@views col_g[1:n] .= ν[k+n] .+ ν[1:n] .+ K[:,k]
			@views col_g[n+1:end] .= ν[k] .+ ν[n+1:end] .+ K[k,:]
			clamp!(col_g, 0, Inf)
			@views G[k+n] = sum(col_g[1:n])
			@views G[k] = sum(col_g[n+1:end])
		end
		G ./= γ
		G .-= 1
	end

	if solver == "LBFGS"
		results = optimize(f, g!, ν0, LBFGS())
	elseif solver == "GradientDescent"
		results = optimize(f, g!, ν0, GradientDescent())
	else
		throw("Invalid solver")
	end
	ν = Optim.minimizer(results)

	if verbose; println("MinMax α: ", minimum(ν[1:n]), " | ", maximum(ν[1:n])); end
	if verbose; println("MinMax β: ", minimum(ν[n+1:end]), " | ", maximum(ν[n+1:end])); end

	C = spzeros(n,n)
	for i in 1:n
		@views C[:,i] .= clamp.(ν[i+n] .+ ν[1:n] .+ K[:,i], 0, Inf)
	end
	C ./= γ
	return C
end


function adssc(X::AbstractMatrix, η1::Number, η2::Number; η3::Number=0.0)
	""" A-DSSC model from 'Doubly Stochastic Subspace Clustering'
	"""
	if η3 < 1e-8 # no l1 regularization
		C = lsr(X, η1; zero_diag=true)
	else
		ensc_gamma = 1/(η1 + η3)
		ensc_tau = η3/(η1 + η3)
		C = ensc(X, ensc_gamma, ensc_tau)
	end
	A = ot_q_dual(abs.(C), η2)
	return A
end




function joint_lsr_linearized_alt(X::AbstractMatrix, γ1::Number, γ2::Number; niters::Int=2000, ρ::Number=.5, ϵ::Number=1e-8, verbose::Bool=true, τ1::Number=0.0001, γ3::Number=0.0)
	""" Main optimizer for J-DSSC model
	"""
	d = size(X, 1)
	n = size(X, 2)
	# initialize variables
	Q = lsr(X, γ1^2; zero_diag=true)
	Qp = zeros(n,n) 
	Qn = zeros(n,n)
	Qn[Q .< 0] .= -Q[Q .< 0]
	Qp[Q .> 0] .= Q[Q .> 0]
	C = ones(n,n)/(n-1)
	C .= C - Diagonal(C)
	A = copy(C)
	Z = zeros(d,n)

	# lagrange multipliers
	λ1 = zeros(n,1)
	λ2 = zeros(n,1)
	Λ1 = zeros(n,n)
	Λ2 = zeros(d,n)

	# intermediate variables
	Qpi = zeros(n,n)
	Qni = zeros(n,n)
	B = zeros(n,n)
	V = zeros(n,n)
	V1 = zeros(n,1)
	V2 = zeros(1,n)
	rowsum_diff = zeros(n,1)
	colsum_diff = zeros(n,1)
	XQ = zeros(d,n)


	function ∇Q!(Qpi::Matrix{Float64}, Qni::Matrix{Float64}, C::Matrix{Float64}, Qp::Matrix{Float64}, Qn::Matrix{Float64})
		# put gradient into Qpi, Qni
		# equiv to Qpi .= -X'*Λ2 .+ ρ .* X'*(X*(Qp .- Qn) .- Z)
		mul!(XQ, X, Qp .- Qn)
		@. XQ .= XQ .- Z
		mul!(Qpi, X', XQ)
		Qpi .= -X'*Λ2 .+ ρ .* Qpi
		@. Qni .= -Qpi
	end

	for i in 1:niters
		# update Q
		∇Q!(Qpi, Qni, C, Qp, Qn)
		# gradient step
		@. Qpi .= Qp .- τ1*Qpi
		@. Qni .= Qn .- τ1*Qni
		# proximal operator
		@. Qpi .= 1/(γ1^2 + 1/τ1) * (1/τ1 .* Qpi .- γ1^2 .* Qn .+ γ1*γ2 .*C .- γ3)
		@. Qni .= 1/(γ1^2 + 1/τ1) * (1/τ1 .* Qni .- γ1^2 .* Qp .+ γ1*γ2 .*C .- γ3)
		Qp .= Qpi
		Qn .= Qni
		clamp!(Qp, 0, Inf)
		Qp .= Qp - Diagonal(Qp)
		clamp!(Qn, 0, Inf)
		Qn .= Qn - Diagonal(Qn)

		# update C
		@. C .= 1/(ρ + γ2^2) * (γ1*γ2*(Qp .+ Qn) .+ ρ .* A .+ Λ1)
		clamp!(C, 0, Inf)

		# update A
		@. V .= ρ*C .+ 2*ρ .- transpose(λ1) .- λ2 .- Λ1
		V1 .= sum(V, dims=2)
		V1 .= 1/(n+1) * (V1 .- 1/(2*n+1) * sum(V1))
		V2 .= sum(V, dims=1)
		V2 .= 1/(n+1) * (V2 .- 1/(2*n+1) * sum(V2))
		@. A .= 1/ρ .* (V .- V1 .- V2)

		rowsum_diff .= transpose(sum(A, dims=1) .- 1)
		colsum_diff .= sum(A, dims=2) .- 1

		# update Z
		# equiv to XQ .= X*(Qp .- Qn)
		mul!(XQ, X, Qp .- Qn)
		Z .= 1/(1+ρ) * (X .- Λ2 .+ ρ*XQ)
				
		
		# dual ascent
		@. λ1 .= λ1 .+ ρ .* rowsum_diff
		@. λ2 .= λ2 .+ ρ .* colsum_diff
		@. Λ1 .= Λ1 .+ ρ .* (A .- C)
		@. Λ2 .= Λ2 .+ ρ .* (Z .- XQ)

		if (maximum(abs.(rowsum_diff)) < ϵ 
			&& maximum(abs.(colsum_diff)) < ϵ 
			&& maximum(abs.(Qpi)) < ϵ
			&& maximum(abs.(Qni)) < ϵ)
			break
		end
	end
	if verbose
		println("Norm A-C: ", norm(A .- C))
		println("Norm Z-X(Qp-Qn): ", norm(Z .- XQ))
		println("MinMax λ1: ", minimum(λ1), " | ", maximum(λ1))
		println("MinMax λ2: ", minimum(λ2), " | ", maximum(λ2))
		println("MinMax Λ1: ", minimum(Λ1), " | ", maximum(Λ1))
		println("MinMax Λ2: ", minimum(Λ2), " | ", maximum(Λ2))
	end

	return C, Qp, Qn
end
