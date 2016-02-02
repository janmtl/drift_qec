using simulate_quantum_memory

# Setup this simulation instance
timestamp  = convert(Int64, floor(time()*1000));
RATE       = float(ARGS[1]);
MAX_T      = convert(Int64, float(ARGS[2]));
NUM_GRID   = convert(Int64, float(ARGS[3]));
NUM_TRIALS = convert(Int64, float( ARGS[4]));

# Setup I/O for target and ffwd distributions
target_filename = string("target-", RATE, "-", MAX_T, "-", timestamp, ".txt")
target_stream   = open(target_filename, "w")
write(target_stream, "id, time, dt, weight\n")
ffwd_filename   = string("ffwd-", RATE, "-", MAX_T, "-", timestamp, ".txt")
ffwd_stream     = open(ffwd_filename, "w")
write(ffwd_stream, "time,dt,weight\n")

print("Simulate Fast Forward testing data\n");
print("==================================\n\n");
print("Starting up simulation...\n");

# Reseed the RNGs
srand(timestamp);
# Setup the [[15, 1, 3]] code
code = RMCode();
print("+ [[15, 1, 3]] code initialized\n")
# Setup a model with error rate and a top-out at MAX_T cycles
model = Model(RATE, NUM_GRID);
print("+ Single-angle dephase model initialized\n")
print("Setup complete.\n\n")

################
# Target model #
################
print("Starting up $NUM_TRIALS trials of `Target Model`...\n")

trials = 0;
runs   = 0;
while trials < NUM_TRIALS

	(time, dt, W_X, W_Z, DT) = single_angle(model, code, MAX_T;
	                                        adaptive=true,
																					simulate_decoder=false,
																					report=true);
	W_X_locs     = [1; nonzerolocs(W_X)];

	wick_lengths = [1; W_X_locs[2:end] - W_X_locs[1:end-1]];
	wick_dts     = DT[W_X_locs];
	wick_weights = sum_to_right_edge(W_Z, W_X_locs);

	for k=1:length(W_X_locs)
		write(target_stream, string(runs));            write(target_stream, ", ");
		write(target_stream, string(wick_lengths[k])); write(target_stream, ", ");
		write(target_stream, string(wick_dts[k]));     write(target_stream, ", ");
		write(target_stream, string(wick_weights[k])); write(target_stream, "\n");
	end
	write(target_stream, "\n");

	trials = trials + length(W_X_locs);
	runs = runs + 1;
end
print("Wrote $trials trials and $runs runs of `Target Model`.\n\n")


################
# Fast Forward #
################
print("Starting up $NUM_TRIALS trials of `Fast Forward`...\n")

for k=1:NUM_TRIALS
  dt = rand()*pi/8;
  (wick_length, spark, wick_weight) = fast_forward(code, model, dt;
																									 update_p_theta = false);
	write(ffwd_stream, string(wick_length)); write(ffwd_stream, ", ");
	write(ffwd_stream, string(dt));          write(ffwd_stream, ", ");
	write(ffwd_stream, string(wick_weight)); write(ffwd_stream, "\n");
end
print("Wrote $trials trials of `Fast Forward`.\n\n");


# Close file I/O
close(target_stream);
close(ffwd_stream);
print("Simulation complete. Results written to:\n");
print("+ Target model: $target_filename\n");
print("+ Fast forward: $ffwd_filename\n");
