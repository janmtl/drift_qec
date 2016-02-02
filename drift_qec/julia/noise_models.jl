module noise_models
    using Symplectic
    using Cubature: hquadrature

    type SingleAngleModel
        n::Int64
        d_x::Int64
        d_z::Int64
        rate::Float64
        theta::Array{Float64, 1}
        grains::Float64
        S::Array{Float64, 1}
        M::Array{Float64, 1}
        p_theta::Array{Float64, 1}
        theta_hat::Float64
        M_hat::Int64
        theta_true::Float64
        M_true::Int64
        sampling_rates::Dict
        update_functions::Dict

        function SingleAngleModel(p::Float64, grains::Number)
            this = new();
            this.n = 15;
            this.d_x = 3;
            this.d_z = 1;
            this.rate = p;
            this.theta = linspace(0, pi/2, grains);
            this.grains = convert(Float64, grains);
            this.S = linspace(-pi, pi, 4*grains+1);
            this.M = 0.5*(this.S[1:end-1] + this.S[2:end]);
            this.p_theta = log(ones(convert(Int64, grains)) ./ grains);
            this.theta_hat = pi/4;
            this.M_hat = floor(grains/2);
            this.theta_true = (0.05+0.9*rand())*pi/2;
            this.M_true = floor(this.theta_true/(pi/2)*grains);

            # SAMPLING RATES
            P_XZ = zeros(this.n+1, this.n+1, length(this.M));
            COSES = cos(this.M - this.theta_true).^2;
            SINES = sin(this.M - this.theta_true).^2;
            RUNNING_COSES = ones(length(COSES));
            RUNNING_SINES = ones(length(SINES));
            COSESK = ones(this.n+1, length(COSES));
            SINESK = ones(this.n+1, length(COSES));

            tic();
            for k=0:this.n
                COSESK[k+1, :] = deepcopy(RUNNING_COSES);
                SINESK[k+1, :] = deepcopy(RUNNING_SINES);
                RUNNING_COSES = RUNNING_COSES .* COSES;
                RUNNING_SINES = RUNNING_SINES .* SINES;
            end
            t = toq();
            # print("marginal arrays time: $t\n");

            tic();
            BINOMS = zeros(this.n+1, this.n+1);
            for j=0:this.n
                for k=0:(this.n-j)
                    BINOMS[j+1, k+1] = binomial(this.n, j) * binomial(this.n-j, k);
                end
            end
            t = toq();
            # print("binomial array time: $t\n");

            tic()
            RATES = ones(this.n+1, this.n+1);
            for j=0:this.n
                for k=0:(this.n-j)
                    RATES[j+1, k+1] = (this.rate)^(j+k) * (1-this.rate)^(this.n - j - k);
                end
            end
            t = toq();
            # print("rates array time: $t\n");

            COEFFICIENTS = BINOMS .* RATES;

            tic();
            for j=0:1:this.n
                for k=0:(this.n-j)
                    P_XZ[j+1, k+1, :] = COEFFICIENTS[j+1, k+1] * COSESK[j+1, :] .* SINESK[k+1, :];
                end
            end
            t = toq();
            # print("joint arrays time: $t\n");

            tic();
            this.sampling_rates = Dict("W_X>0" => reshape(sum(P_XZ[2:end, :,  :], [1, 2]), length(this.M)),
                                       "W_Z>0" => reshape(sum(P_XZ[:, 2:end, :], [1, 2]), length(this.M)),
                                       "W_X>d_x" => reshape(sum(P_XZ[this.d_x+2:end, :, :], [1, 2]), length(this.M)),
                                       "W_X=k|d_x>=W_X>0" => reshape(sum(P_XZ[2:this.d_x+1, :, :], [2]) ./ sum(P_XZ[2:this.d_x+1, :, :], [1, 2]), this.d_x, length(this.M)),
                                       "W_Z>d_z|W_Z>0" => reshape(sum(P_XZ[:, this.d_z+2:end, :], [1, 2]) ./ sum(P_XZ[:, 2:end, :], [1, 2]), length(this.M)),
                                       "uncorrectable" => reshape(sum(P_XZ[1:this.d_x+1, this.d_z+2:end, :], [1, 2])
                                                           + sum(P_XZ[this.d_x+2:end, 1:this.d_z+1, :], [1, 2])
                                                           + sum(P_XZ[this.d_x+2:end, this.d_z+2:end, :], [1, 2]), length(this.M))
                                    )
            t = toq();
            # print("summations time: $t\n");

            # UPDATE FUNCTIONS
            # Z=1, X=3, p^4 (5x/128 + 1/64*sin(2x) - 1/128 *sin(4x) - 1/192*sin(6x) - 1/1024*sin(8x)) * (1-p)^11
            # Z=1, X=2, p^3 (x/16 + 1/64*sin(2x) - 1/64*sin(4x) - 1/192*sin(6x)) * (1-p)^12
            # Z=1, X=1, p^2 (x/8 - 1/32*sin(4x)) * (1-p)^13
            # Z=0, X=3, p^3 (5x/16 + 15/64*sin(2x) + 3/64*sin(4x) + 1/192*sin(6x)) * (1-p)^12
            # Z=0, X=2, p^2 (3x/8 + 1/4*sin(2x) + 1/32*sin(4x)) * (1-p)^13
            # Z=0, X=1, p^1 (x/2 + 1/4*sin(2x)) * (1-p)^14

            tic();
            p = this.rate;
            PL = Dict("Z=1, X=0" => x -> p^1 * (x/2 - 1/4*sin(2*x)) * (1-p)^14,
                      # "Z=1, X=1" => x -> p^2 * (x/8 - 1/32*sin(4*x)) * (1-p)^13,
                      # "Z=1, X=2" => x -> p^3 * (x/16 + 1/64*sin(2*x) - 1/64*sin(4*x) - 1/192*sin(6*x)) * (1-p)^12,
                      # "Z=1, X=3" => x -> p^4 * (5*x/128 + 1/64*sin(2*x) - 1/128 *sin(4*x) - 1/192*sin(6*x) - 1/1024*sin(8*x)) * (1-p)^11,
                      "Z=0, X=1" => x -> p^1 * (x/2 + 1/4*sin(2*x)) * (1-p)^14,
                      # "Z=0, X=2" => x -> p^2 * (3*x/8 + 1/4*sin(2*x) + 1/32*sin(4*x)) * (1-p)^13,
                      # "Z=0, X=3" => x -> p^3 * (5*x/16 + 15/64*sin(2*x) + 3/64*sin(4*x) + 1/192*sin(6*x)) * (1-p)^12
                    )

            this.update_functions = Dict("Z=1, X=0" => log(PL["Z=1, X=0"](this.S[2:end]) - PL["Z=1, X=0"](this.S[1:end-1])),
                                         # "Z=1, X=1" => PL["Z=1, X=1"](this.S[2:end]) - PL["Z=1, X=1"](this.S[1:end-1]),
                                         # "Z=1, X=2" => PL["Z=1, X=2"](this.S[2:end]) - PL["Z=1, X=2"](this.S[1:end-1]),
                                         # "Z=1, X=3" => PL["Z=1, X=3"](this.S[2:end]) - PL["Z=1, X=3"](this.S[1:end-1]),
                                         "Z=0, X=1" => log(PL["Z=0, X=1"](this.S[2:end]) - PL["Z=0, X=1"](this.S[1:end-1])),
                                         # "Z=0, X=2" => PL["Z=0, X=2"](this.S[2:end]) - PL["Z=0, X=2"](this.S[1:end-1]),
                                         # "Z=0, X=3" => PL["Z=0, X=3"](this.S[2:end]) - PL["Z=0, X=3"](this.S[1:end-1])
                                        )

            update_functions_profile = toq();
            # print("Update functions time: $update_functions_profile\n")
            return this
        end
    end
    export SingleAngleModel

    function rand_error(model::SingleAngleModel, n::Int64, dt::Float64)
        error_types = (rand(1,n) .< sin(dt)^2);
        error_sites = (rand(1,n) .< model.rate);
        symp = Symp((error_sites & error_types), (error_sites & ~error_types));
        return symp
    end
    export rand_error

    function update(model::SingleAngleModel, err::Symp, theta_est::Float64)
        w_x = convert(Float64, sum(err.x));
        w_z = convert(Float64, sum(err.z));
        c_t = cos(model.theta - theta_est).^(2*w_z);
        s_t = sin(model.theta - theta_est).^(2*w_x);
        p_t_given_err = c_t .* s_t .* model.p_theta;
        p_t_given_err = p_t_given_err / sum(p_t_given_err);
        model.p_theta = p_t_given_err;
        return model
    end
    export update

end
