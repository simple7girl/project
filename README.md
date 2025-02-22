# project
% Clear Workspace
clear; clc; close all;

%% 1. Basic IRSA-NOMA Simulation with Graphical Outputs
% Parameters for Basic IRSA-NOMA Simulation
numDevices = 100;             % Total IoT devices
emergencyDevices = 10;        % Number of high-priority devices
slots = 50;                   % Number of slots available
simulationRounds = 100;       % Number of simulation rounds
trafficIntensity = 5;         % Average traffic load per device (Poisson parameter)

% Initialize performance metrics
throughput = zeros(1, simulationRounds);
emergencySuccessRate = zeros(1, simulationRounds);
averageLatency = zeros(1, simulationRounds);

% Simulation Loop for IRSA-NOMA
for round = 1:simulationRounds
    % Generate Traffic Load (Poisson distribution)
    trafficLoad = poissrnd(trafficIntensity, numDevices, 1);

    % Assign Priorities (1 for emergency devices, 0 for normal)
    priority = zeros(numDevices, 1);
    priority(1:emergencyDevices) = 1;

    % Adaptive Slot Allocation
    activeDevices = sum(trafficLoad > 0);
    slotsAllocated = min(slots, ceil(activeDevices * 1.5));

    % Simulate IRSA-NOMA Transmission
    successfulTransmissions = 0;
    emergencyTransmissions = 0;
    totalDelay = 0;

    for device = 1:numDevices
        if trafficLoad(device) > 0
            % Calculate repetitions (higher priority gets more repetitions)
            repetitions = 1 + priority(device) * 2; % Emergency devices get 3 repetitions
            successful = min(repetitions, slotsAllocated / activeDevices);

            % Count successful transmissions
            successfulTransmissions = successfulTransmissions + successful;
            if priority(device) == 1
                emergencyTransmissions = emergencyTransmissions + successful;
            end

            % Estimate delay (inverse proportional to success rate)
            delay = 1 / (successful + 1e-6);
            totalDelay = totalDelay + delay;
        end
    end

    % Store metrics
    throughput(round) = successfulTransmissions / slotsAllocated;
    emergencySuccessRate(round) = emergencyTransmissions / emergencyDevices;
    averageLatency(round) = totalDelay / activeDevices;
end

% Plot IRSA-NOMA Simulation Results
figure;

% Throughput Graph
subplot(3, 1, 1);
plot(1:simulationRounds, throughput, 'b-o', 'LineWidth', 1.5);
title('Network Throughput');
xlabel('Simulation Round');
ylabel('Throughput');
grid on;

% Emergency Success Rate Graph
subplot(3, 1, 2);
plot(1:simulationRounds, emergencySuccessRate, 'r-*', 'LineWidth', 1.5);
title('Emergency Success Rate');
xlabel('Simulation Round');
ylabel('Success Rate');
grid on;

% Average Latency Graph
subplot(3, 1, 3);
plot(1:simulationRounds, averageLatency, 'g-s', 'LineWidth', 1.5);
title('Average Latency');
xlabel('Simulation Round');
ylabel('Latency');
grid on;

% Display Results for IRSA-NOMA
disp('IRSA-NOMA Simulation Complete!');

%% 2. Advanced System Simulation with Energy Efficiency, Traffic Management, etc.
% Parameters for Advanced System
num_devices = 100; % Total IoT devices
priority_levels = randi([1, 3], 1, num_devices); % Priority levels (1-High, 2-Medium, 3-Low)
access_probs = [0.9, 0.6, 0.3]; % Access probabilities based on priority
energy_levels = rand(1, num_devices) * 100; % Initial energy in Joules
tx_energy = 1.5; % Energy consumed per transmission in Joules
traffic_data = rand(num_devices, 1); % Simulated traffic intensity
edge_server_capacity = 1000; % Edge server processing units
freq = 300e9; % 300 GHz for THz communication
bandwidth = 10e9; % 10 GHz bandwidth
distance = 100; % Communication distance in meters

% Energy Efficiency Simulation
remaining_energy = zeros(1, num_devices); % Track remaining energy
for device = 1:num_devices
    if energy_levels(device) > tx_energy
        energy_levels(device) = energy_levels(device) - tx_energy;
    end
    remaining_energy(device) = energy_levels(device);
end

% Plot Remaining Energy
figure;
bar(1:num_devices, remaining_energy);
title('Remaining Energy Levels of Devices');
xlabel('Device Index');
ylabel('Remaining Energy (Joules)');
grid on;

%% Multi-Priority Traffic Management
priority_counts = histcounts(priority_levels, 1:4); % Count devices by priority
% Plot Priority Distribution
figure;
pie(priority_counts, {'High', 'Medium', 'Low'});
title('Priority Distribution of Devices');

%% AI-Driven Traffic Prediction
mdl = fitrtree(traffic_data, access_probs(priority_levels)'); % Train regression tree model
new_traffic = linspace(0, 1, 50)'; % Simulated traffic intensities
predicted_probs = predict(mdl, new_traffic);

% Plot Predicted Access Probabilities
figure;
plot(new_traffic, predicted_probs, 'LineWidth', 2);
title('Predicted Access Probability vs Traffic Intensity');
xlabel('Traffic Intensity');
ylabel('Access Probability');
grid on;

%% Edge Computing Simulation
traffic_load = randi([10, 50], 1, num_devices); % Traffic load per device
processed_traffic = min(edge_server_capacity, sum(traffic_load));
remaining_traffic = sum(traffic_load) - processed_traffic;

% Pie Chart of Traffic Load Processing
figure;
pie([processed_traffic, remaining_traffic], {'Processed', 'Unprocessed'});
title('Edge Server Traffic Load Distribution');

%% Successive Interference Cancellation (SIC)
signal_strengths = rand(1, num_devices); % Random signal strengths
threshold = 0.2; % Interference threshold
decoded_signals = signal_strengths(signal_strengths > threshold);
undecoded_signals = signal_strengths(signal_strengths <= threshold);

% Plot Signal Decoding Results
figure;
histogram(decoded_signals, 'FaceColor', 'g', 'BinWidth', 0.05); hold on;
histogram(undecoded_signals, 'FaceColor', 'r', 'BinWidth', 0.05);
title('Signal Strength Distribution (Decoded vs Undecoded)');
xlabel('Signal Strength');
ylabel('Number of Devices');
legend('Decoded Signals', 'Undecoded Signals');
grid on;

%% Beyond 6G Communication (THz Channel)
path_loss = 20 * log10(freq) + 20 * log10(distance) - 147.55; % Path loss in dB

% Bar Chart of Path Loss
figure;
bar(path_loss, 'FaceColor', [0.2, 0.4, 0.8]);
title('Path Loss for Terahertz Communication');
ylabel('Path Loss (dB)');
set(gca, 'XTickLabel', {'300 GHz Frequency'});
grid on;

%% End of Advanced Simulation
disp('Advanced System Simulation Completed Successfully!');
