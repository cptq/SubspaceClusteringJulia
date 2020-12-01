using LinearAlgebra
using MAT

function gen_synthetic(d::Int, sub_d::Int, nspaces::Int, npoints::Int, noise::Number; orthogonal::Bool=false)
	"""
	Translated from https://github.com/ChongYou/subspace-clustering

	This function generates a union of subspaces under random model, i.e., 
    subspaces are independently and uniformly distributed in the ambient space,
    data points are independently and uniformly distributed on the unit sphere of each subspace

    Parameters
    -----------
    d : int
        Dimension of the ambient space
    sub_d : int
        Dimension of each subspace (all subspaces have the same dimension)
    nspaces : int
        Number of subspaces to be generated
    npoints : int
        Number of data points from each of the subspaces
    noise : float
        Amount of Gaussian noise on data
    Returns
    -------
    data : shape d × (nspaces * npoints)
        Data matrix containing points drawn from a union of subspaces as its rows
    label : shape (nspaces * npoints)
        Membership of each data point to the subspace it lies in
    """
	data = zeros(d, nspaces * npoints)
	label = zeros(Int, nspaces*npoints)

	if orthogonal
		@assert sub_d * nspaces <= d "Subspaces too high dimensional to be orthogonal"
		
		bases = qr(randn(d, sub_d*nspaces)).Q
		for i in 1:nspaces
			sub_inds = 1+(i-1)*sub_d:i*sub_d
			basis = bases[:,sub_inds]
			coeff = mapslices(normalize, randn(sub_d, npoints); dims=1)
			samples = basis * coeff
			samples += randn(d, npoints)*noise
			samples = mapslices(normalize, samples; dims=1)
			inds = 1+(i-1)*npoints:i*npoints
			data[:, inds] = samples
			label[inds] .= i
		end

		return data, label
	end

	for i in 1:nspaces
		basis = qr(randn(d, sub_d)).Q
		coeff = mapslices(normalize, randn(sub_d, npoints); dims=1)
		samples = basis * coeff
		samples += randn(d, npoints)*noise
		samples = mapslices(normalize, samples; dims=1)
		inds = 1+(i-1)*npoints:i*npoints
		data[:, inds] = samples
		label[inds] .= i
	end

	return data, label
end

function load_umist_resized()
	""" Loads UMIST face dataset """
	nspaces = 20
    pathname = "data/umist_resized.mat"
    vars = matread(pathname)
    X = vars["X"]
    X = Matrix{Float64}(X)
    label = vec(vars["label"])
    label = Array{Int}(label)
    norm_X = mapslices(normalize, X, dims=1)
    return norm_X, nspaces, label
end

function load_scattered_coil(;nspaces=100)
	""" Loads scattered COIL dataset """
    pathname = "data/COIL_scatter.mat"
    vars = matread(pathname)
    X = vars["data"]
    X = Matrix{Float64}(X)
    label = vec(vars["label"])
    label = Array{Int}(label)
    norm_X = mapslices(normalize, X, dims=1)
    norm_X = norm_X[:,1:nspaces*72]
    label = label[1:nspaces*72];
    return norm_X, nspaces, label
end

function load_scattered_umist()
	""" Loads scattered UMIST dataset """
    nspaces = 20
    pathname = "data/umist_scatter.mat"
    vars = matread(pathname)
    X = vars["data"]
    X = Matrix{Float64}(X)
    label = vec(vars["label"])
    label = Array{Int}(label)
    norm_X = mapslices(normalize, X, dims=1)
    return norm_X, nspaces, label
end
