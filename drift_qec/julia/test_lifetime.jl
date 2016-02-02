using simulate_quantum_memory
code = RMCode();

# File I/O
timestamp  = convert(Int64, floor(time()*1000));
salt = getpid();
f = open("ffwd-$salt-$timestamp.csv", "w");

# Settings
NUM_TRIALS_PER_RATE = 100;
RATES = logspace(-2, -6, 10);
MODELGRAINS = ceil(5 ./ RATES);
MAX_TS = RATES .^ (-4);

write(f, "# NUM_TRIALS_PER_RATE = $NUM_TRIALS_PER_RATE\n");

# Trials
write(f, "rate, modelgrain, max_t, time, theta_hat, theta_true, exit_status, p_uncorrectable\n")
close(f);
for j=1:length(RATES)
    rate = RATES[j];
    modelgrain = MODELGRAINS[j];
    max_t = MAX_TS[j];
    print("Simulating $NUM_TRIALS_PER_RATE trials of rate = $rate\n")
    for k=1:NUM_TRIALS_PER_RATE
        model = SingleAngleModel(rate, modelgrain);
        EXPORT = single_angle_fast_forward(model, max_t, report=false)
        time = EXPORT["time"];
        exit_status = EXPORT["exit_status"];
        p_uncorrectable = EXPORT["p_uncorrectable"];
        theta_hat = model.theta_hat;
        theta_true = model.theta_true;
        print("$rate, $modelgrain, $max_t, $time, $theta_hat, $theta_true, $exit_status, $p_uncorrectable\n")
        f = open("ffwd-$salt-$timestamp.csv", "a");
        write(f, "$rate, $modelgrain, $max_t, $time, $theta_hat, $theta_true, $exit_status, $p_uncorrectable\n");
        close(f);
    end
end
f = open("ffwd-$salt-$timestamp.csv", "a");
write(f, "# EOF");
close(f);
