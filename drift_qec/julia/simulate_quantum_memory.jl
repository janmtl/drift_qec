module simulate_quantum_memory
  using Symplectic
  using css_code
  using noise_models
  export SingleAngleModel
  using Distributions: Geometric, Categorical, Bernoulli, Normal


  # Utility functions
  function changelocs(arr)
    n = size(arr)[1];
    locs = collect(1:n-1)[(arr[2:n]-arr[1:n-1] .!= 0)];
    return locs
  end
  function nonzerolocs(arr)
    n = size(arr)[1];
    locs = collect(1:n)[(arr .!= 0)];
    return locs
  end
  function sum_to_right_edge(arr, edges)
    sums = zeros(length(edges));
    edges = [1; edges];
    for k=1:length(sums)
        sums[k] = sum(arr[edges[k]:edges[k+1]]);
    end
    return sums
  end
  export changelocs, nonzerolocs, sum_to_right_edge


################################################################################
##                     [[15, 1, 3]] Asymmetric CSS Code                       ##
################################################################################

    function RMCode()
        # The bitstrings that describe each stabilizer operator
        x_S=[[0 0 0 0 0 0 0 1 1 1 1 1 1 1 1];
             [0 0 0 1 1 1 1 0 0 0 0 1 1 1 1];
             [0 1 1 0 0 1 1 0 0 1 1 0 0 1 1];
             [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1]];
        z_S=[[0 0 0 0 0 0 0 1 1 1 1 1 1 1 1];
             [0 0 0 1 1 1 1 0 0 0 0 1 1 1 1];
             [0 1 1 0 0 1 1 0 0 1 1 0 0 1 1];
             [1 0 1 0 1 0 1 0 1 0 1 0 1 0 1];
             [1 0 0 0 0 0 1 0 0 1 0 1 0 0 0];
             [0 1 0 0 0 0 1 0 1 0 0 1 0 0 0];
             [0 0 1 0 0 0 1 1 0 0 0 1 0 0 0];
             [0 0 0 1 0 0 1 0 1 1 0 0 0 0 0];
             [0 0 0 0 1 0 1 1 0 1 0 0 0 0 0];
             [0 0 0 0 0 1 1 1 1 0 0 0 0 0 0]];

        # The same operators, now in symplectic form
        X_S = Symp(zeros(size(x_S)), (x_S .== 1));
        Z_S = Symp((z_S .== 1), zeros(size(z_S)));
        code = Code(1, 3, X_S, Z_S)
        return code
    end
    export RMCode


################################################################################
##                 Single-angle dephasing noise simulation                    ##
################################################################################

    function single_angle_full_simulation(model::SingleAngleModel,
                                          code::Code,
                                          MAX_T::Number;
                                          simulate_decoder=simulate_decoder::Bool,
                                          report=report::Bool)

     #########################
     #         Setup         #
     #########################

     theta_true = (0.05+0.9*rand())*pi/2;
     theta_est = pi/4;
     time = 0;
     if report == true
       W_X = [0.0];
       W_Z = [0.0];
       T0s = [0.0];
       TEs = [0.0];
       TIMEs = [0.0];
       report_idx = 1;
     end

     #########################
     #       Simulation      #
     #########################

     for T=1:MAX_T
         original_error = rand_error(model, code.n, theta_true - theta_est);
         w_x = sum(original_error.x); w_z = sum(original_error.z);

         if simulate_decoder == true
           (s_x, s_z) = code.measure(original_error);
           decoded_error, flag = code.decode(s_x, s_z);
         else
           decoded_error, flag = original_error, convert(Int64, (w_x <= 1) & (w_z <=3))
         end

         if flag != 1
             break
         else
             time = time + 1
         end

         model = update(model, decoded_error, theta_est);
         theta_est = deepcopy(model.theta[indmax(model.p_theta)]);

         ## REPORTING
         if report == true
           push!(W_X, w_x);
           push!(W_Z, w_z);
           push!(T0s, theta_true);
           push!(TEs, theta_est);
           push!(TIMEs, time);
           report_idx = report_idx + 1;
         end
     end

     #########################
     #         Report        #
     #########################

     if report == false
       return (time, abs(theta_true - theta_est))
     else
       return (TIMEs, T0s, TEs, W_X, W_Z)
     end

    end
    export single_angle_full_simulation


################################################################################
##           Single-angle dephasing noise fast-forward simulation             ##
################################################################################

    function single_angle_fast_forward(model::SingleAngleModel,
                                       MAXTIME::Number;
                                       report=report::Bool)

        #########################
        #         Setup         #
        #########################
        time = 0;
        EXPORT = Dict("time" => 0.0, "exit_status" => 0.0, "p_uncorrectable" => 0.0)
        if report == true
            REPORT = Dict("W_X" => [0.0],
                          "W_Z" => [0.0],
                          "T0s" => [0.0],
                          "TEs" => [0.0],
                          "TIMEs" => [0.0],
                          "CYCLE_TYPEs" => [0],
                          "PROFILEs" => [0.0],
                          "M_HATs" => [0],
                          "M_TRUEs" => [0]);
        end

        #########################
        #       Simulation      #
        #########################
        uninformative_continue = true;
        informative_continue = true;
        exit_status = 0;
        while (time < MAXTIME) & uninformative_continue & informative_continue

            idx = convert(Int64, 2*model.grains + model.M_hat);

            # Keep simulating plateaus

            ## SAMPLE TIME
            # Sample the amount time until the next informative error
            p_inf = model.sampling_rates["W_Z>0"][idx]
            T_inf = Geometric(p_inf);
            t_inf = rand(T_inf);

            # Sample uncorrectable uninformative errors that occured during t
            f_uninf = -randexp() / log1p(-model.sampling_rates["W_X>d_x"][idx]);

            # Test for failure
            uninformative_continue = (t_inf < f_uninf);


            ## SAMPLE INFORMATIVE ERROR
            # Sample the weight of the informative error
            M_inf = Bernoulli(model.sampling_rates["W_Z>d_z|W_Z>0"][idx]);
            m_inf = rand(M_inf)+1;

            # Test for failure
            informative_continue = (m_inf <= model.d_z);


            ## SAMPLE UNINFORMATIVE ERROR
            # Sample the number of uninformative errors that have occured up to t
            n_uninf = 0;
            p_uninf = model.sampling_rates["W_X>0"][idx];
            p_cat = model.sampling_rates["W_X=k|d_x>=W_X>0"][:, idx]
            if t_inf * p_uninf < 5
                tic();
                cycle_type=0;
                t_skip = 0;
                Et_uninf = 1/p_uninf;
                T_uninf = Geometric(p_uninf);
                while (t_skip <= t_inf)
                    t_uninf = rand(T_uninf);
                    n_uninf = n_uninf + 1;
                    t_skip = t_skip + t_uninf;
                end
                M_uninf = Categorical(p_cat);
                m_uninf = rand(M_uninf, n_uninf);
                m_uninfs = hist(m_uninf,0.5:model.d_x+0.5)[2];
                profile = toq();
            else
                tic();
                N_uninf = Normal(t_inf*p_uninf, t_inf*p_uninf*(1-p_uninf));
                n_uninf = -1;
                while n_uninf < 0
                    n_uninf = convert(Int64, floor(rand(N_uninf)));
                end
                if n_uninf > 1 / p_cat[2]
                    cycle_type=2;
                    m_uninfs = convert(Array{Int64, 1}, floor(n_uninf .* p_cat));
                else
                    cycle_type=1;
                    M_uninf = Categorical(p_cat);
                    m_uninf = rand(M_uninf, n_uninf);
                    m_uninfs = hist(m_uninf,0.5:model.d_x+0.5)[2];
                end
                profile = toq();
            end

            ## UPDATE MODEL
            # Informative update
            time = copy(time + t_inf)

            # Update the probability distribution based on the informative error
            idx = convert(Int64, 2*model.grains-model.M_hat);
            g = convert(Int64, model.grains-1);
            update = model.update_functions["Z=1, X=0"][idx:idx+g]';

            # Uninformative update
            for k=1:model.d_x
                update = update + m_uninfs[k] * model.update_functions["Z=0, X=1"][idx:idx+g]'
            end

            model.p_theta = model.p_theta + update[:];
            model.p_theta = model.p_theta - mean(model.p_theta);

            # Update the MLE
            model.M_hat = indmax(model.p_theta);
            idx_hat = convert(Int64, 2*model.grains + model.M_hat);
            model.theta_hat = copy(model.M[idx_hat]);

            M_d = model.M_hat - model.M_true;
            p_uncorr = model.sampling_rates["uncorrectable"][idx_hat]

            ## REPORTING
            if report == true
                print("$M_d, $p_inf, $p_uncorr, $t_inf\n");
                push!(REPORT["W_X"], sum(m_uninfs));
                push!(REPORT["W_Z"], m_inf);
                push!(REPORT["T0s"], model.theta_true);
                push!(REPORT["TEs"], model.theta_hat);
                push!(REPORT["TIMEs"], time);
                push!(REPORT["CYCLE_TYPEs"], cycle_type);
                push!(REPORT["PROFILEs"], profile);
                push!(REPORT["M_HATs"], model.M_hat);
                push!(REPORT["M_TRUEs"], model.M_true);
            end

            # EXITING THE LOOP
            if !informative_continue
                exit_status = 0 # informative_overweight
            elseif !uninformative_continue
                exit_status = 1 # uninformative_overweight
            elseif !(time < MAXTIME)
                exit_status = 2 # time_out
            end
        end

        #########################
        #         Report        #
        #########################

        EXPORT["time"] = time;
        EXPORT["exit_status"] = exit_status;
        idx_hat = convert(Int64, 2*model.grains + model.M_hat);
        EXPORT["p_uncorrectable"] = model.sampling_rates["uncorrectable"][idx_hat];

        if report == false
            return EXPORT
        else
            return (EXPORT, REPORT)
        end
  end
  export single_angle_fast_forward
end
