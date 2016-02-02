module errors

	function _next_perm(v::Int64)
	    v = unsigned(v);
	    t = (v | unsigned(v - 1)) + 1;
	    t = unsigned(t);
	    w = t | ((div((t & -t), (v & -v)) >> 1) - 1);
	    return convert(Int64, w)
	end

	function _all_bitstrings_of_weight(n, w)
		K = binomial(n, w);
		perm = 2^w-1;
		perms = [];
		for k=1:K
		    perms = [perms; perm];
		    perm = _next_perm(perm);
		end

		E = zeros(n, K);
		for k=1:K
		    E[:, k] = digits(perms[k], 2, n)
		end
		return convert(Array{Bool, 2}, E)
	end


	function _all_bitstrings_up_to_weight(n, w)
		w_0 = convert(Array{Bool, 2}, zeros(n, 1))
		w_1 = convert(Array{Bool, 2}, eye(n))
		# All errors up to weight w_x
		if w > 1
			# All weight 1 errors
			out = [w_0 w_1]
			# All other weights
			for j=2:w
				out = [out _all_bitstrings_of_weight(n, j)]
			end
		elseif w == 1
			out = [w_0 w_1]
		elseif w == 0
			out = w_0
		end
		return out
	end

	# Generate all errors up to and including weights w_x and w_z
	function _errors(n, w_x, w_z)
		# All X errors up to weight w_x
		out_x = _all_bitstrings_up_to_weight(n, w_x)'

		# All Z errors up to weight w_z
		out_z = _all_bitstrings_up_to_weight(n, w_z)'

		# Take the outer product of the errors
		n_x = size(out_z)[1];
		n_z = size(out_x)[1];
		rep_x = repeat(out_x, inner=[n_x; 1]);
		rep_z = repeat(out_z, outer=[n_z; 1]);
		return (rep_x, rep_z)
	end
	export _errors, _all_bitstrings_up_to_weight, _all_bitstrings_of_weight

end
