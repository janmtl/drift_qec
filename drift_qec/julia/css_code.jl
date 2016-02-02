module css_code
	using Symplectic
	using errors

	type Code
		n::Int64
		w_x::Int64
		w_z::Int64
		X_S::Symp
		Z_S::Symp
		S_X::Array{Int64, 1}
		S_Z::Array{Int64, 1}
		E::Symp

		measure::Function
		decode::Function

		function Code(w_x::Int64, w_z::Int64, X_S::Symp, Z_S::Symp)

			this = new()

      		this.n = size(X_S)[2];
			this.w_x = w_x;
			this.w_z = w_z;
			this.X_S = X_S;
			this.Z_S = Z_S;

			(e_x, e_z) = _errors(this.n, this.w_x, this.w_z)
			n_E = size(e_x)[1];
			this.E = Symp(e_x, e_z)

			d_x = 2.^(0:(size(this.X_S)[1]-1));
			d_z = 2.^(0:(size(this.Z_S)[1]-1));
			this.S_X = zeros(n_E);
			this.S_Z = zeros(n_E);
			for k = 1:n_E
				this.S_X[k] = ((this.X_S*this.E[k,:])*d_x)[1];
				this.S_Z[k] = ((this.Z_S*this.E[k,:])*d_z)[1];
			end

			# Measure the syndromes
			this.measure = function(e::Symp)
			    s_x = this.X_S*e;
			    s_z = this.Z_S*e;
					d_x = 2.^(0:(size(s_x)[2]-1));
					d_z = 2.^(0:(size(s_z)[2]-1));
			    return (s_x*d_x, s_z*d_z)
			end

			this.decode = function(s_x::Array{Int64, 1}, s_z::Array{Int64, 1})
				hits_x = (s_x .== this.S_X)
				hits_z = (s_z .== this.S_Z)
				hits = hits_x & hits_z
				err_idx = find(hits)
				if length(err_idx) > 0
					# Correctable error or confusing error
					return (this.E[err_idx[1], :], length(err_idx))
				else
					# Uncorrectable error
					return (nothing, length(err_idx))
				end
			end

			return this
		end
	end
	export Code
end
