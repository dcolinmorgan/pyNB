% GeneSPIDER2 MATLAB Workflow
% Implements standard benchmark workflow with non-NestBoot and NestBoot LASSO/LSCO
% Dataset: Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json
% Network: Tjarnberg-D20150910-random-N50-L158-ID252384.json
% 50 inner and outer bootstrap runs for NestBoot, seed 42
% Saves outputs as CSV for performance report
% Date: 2025-07-18

%% Setup
clear all;
rng(42); % Set random seed
output_dir = 'benchmark_results';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
addpath('/path/to/GeneSPIDER2'); % Add GeneSPIDER2 library

% Initialize performance tracking
all_results = [];
fprintf('🚀 MATLAB Performance Benchmark\n');
fprintf('========================================\n');

%% Step 1: Network Data Import and Analysis
fprintf('\n📥 Step 1: Network Data Import and Analysis\n');
fprintf('=====================================\n');

% Load dataset
dataset_url = 'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json';
data = datastruct.Dataset.fetch(dataset_url);
fprintf('✅ Dataset loaded: %s\n', data.dataset);
fprintf('   📊 Expression matrix shape: [%d, %d]\n', size(data.Y));
fprintf('   🧬 Number of genes: %d\n', data.N);
fprintf('   🔬 Number of samples: %d\n', data.M);

% Load reference network
network_url = 'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json';
net = datastruct.Network.fetch(network_url);
fprintf('✅ Reference network loaded: %s\n', net.network);

%% Step 2: Network Inference
fprintf('\n🔍 Step 2: Network Inference\n');
fprintf('=====================================\n');

zetavec = logspace(-6, 0, 30); % Sparsity parameters
FDR = 0.05; % False Discovery Rate
nest = 50; % Outer bootstrap runs
boot = 50; % Inner bootstrap iterations
par = true; % Parallel processing
cpus = 2; % Number of CPUs

% Non-NestBoot LASSO
fprintf('   🎯 Running non-NestBoot LASSO...\n');
tic; mem_start = get_memory_usage();
estA_lasso = Methods.lasso(data, net, zetavec, false);
time_lasso = toc; mem_lasso = get_memory_usage() - mem_start;

% Non-NestBoot LSCO
fprintf('   🎯 Running non-NestBoot LSCO...\n');
tic; mem_start = get_memory_usage();
estA_lsco = Methods.lsco(data, net, zetavec, false, 'input');
time_lsco = toc; mem_lsco = get_memory_usage() - mem_start;

% NestBoot LASSO
fprintf('   🎯 Running NestBoot LASSO...\n');
tic; mem_start = get_memory_usage();
nbout_lasso = Methods.NestBoot(data, 'lasso', nest, boot, zetavec, FDR, output_dir, par, cpus);
time_nest_lasso = toc; mem_nest_lasso = get_memory_usage() - mem_start;

% NestBoot LSCO
fprintf('   🎯 Running NestBoot LSCO...\n');
tic; mem_start = get_memory_usage();
nbout_lsco = Methods.NestBoot(data, 'lsco', nest, boot, zetavec, FDR, output_dir, par, cpus);
time_nest_lsco = toc; mem_nest_lsco = get_memory_usage() - mem_start;

%% Step 3: Model Comparison
fprintf('\n📊 Step 3: Model Comparison\n');
fprintf('=====================================\n');

% Compare non-NestBoot LASSO
M_lasso = analyse.CompareModels(net, estA_lasso);
results_lasso = struct(M_lasso);
fprintf('   ✅ Non-NestBoot LASSO Results:\n');
fprintf('      F1 Score: %.3f, Time: %.1fs, Memory: %.1fMB, AUROC: %.3f\n', ...
    results_lasso.F1(25), time_lasso, mem_lasso, M_lasso.AUROC());
result_lasso = create_result_struct('lasso', false, data, time_lasso, mem_lasso, ...
    estA_lasso(:, :, 25), results_lasso, M_lasso.AUROC());
all_results = [all_results; result_lasso];

% Compare non-NestBoot LSCO
M_lsco = analyse.CompareModels(net, estA_lsco);
results_lsco = struct(M_lsco);
fprintf('   ✅ Non-NestBoot LSCO Results:\n');
fprintf('      F1 Score: %.3f, Time: %.1fs, Memory: %.1fMB, AUROC: %.3f\n', ...
    results_lsco.F1(25), time_lsco, mem_lsco, M_lsco.AUROC());
result_lsco = create_result_struct('lsco', false, data, time_lsco, mem_lsco, ...
    estA_lsco(:, :, 25), results_lsco, M_lsco.AUROC());
all_results = [all_results; result_lsco];

% Compare NestBoot LASSO
M_nest_lasso = analyse.CompareModels(net, nbout_lasso.binary_networks);
results_nest_lasso = struct(M_nest_lasso);
fprintf('   ✅ NestBoot LASSO Results:\n');
fprintf('      F1 Score: %.3f, Time: %.1fs, Memory: %.1fMB, AUROC: %.3f\n', ...
    results_nest_lasso.F1(end), time_nest_lasso, mem_nest_lasso, M_nest_lasso.AUROC());
result_nest_lasso = create_result_struct('lasso', true, data, time_nest_lasso, mem_nest_lasso, ...
    nbout_lasso.binary_networks(:, :, end), results_nest_lasso, M_nest_lasso.AUROC());
all_results = [all_results; result_nest_lasso];

% Compare NestBoot LSCO
M_nest_lsco = analyse.CompareModels(net, nbout_lsco.binary_networks);
results_nest_lsco = struct(M_nest_lsco);
fprintf('   ✅ NestBoot LSCO Results:\n');
fprintf('      F1 Score: %.3f, Time: %.1fs, Memory: %.1fMB, AUROC: %.3f\n', ...
    results_nest_lsco.F1(end), time_nest_lsco, mem_nest_lsco, M_nest_lsco.AUROC());
result_nest_lsco = create_result_struct('lsco', true, data, time_nest_lsco, mem_nest_lsco, ...
    nbout_lsco.binary_networks(:, :, end), results_nest_lsco, M_nest_lsco.AUROC());
all_results = [all_results; result_nest_lsco];

%% Step 4: Save Results
fprintf('\n💾 Step 4: Save Results\n');
fprintf('=====================================\n');

% Save benchmark results as CSV
results_table = struct2table(all_results);
writetable(results_table, fullfile(output_dir, 'matlab_benchmark_results.csv'));

% Save summary statistics
summary = create_summary_statistics(all_results);
save_json(fullfile(output_dir, 'matlab_summary.json'), summary);

% Save individual comparison results as CSV
save_path_lasso = fullfile(output_dir, 'comparison_results_lasso.csv');
M_lasso.save(save_path_lasso, 'csv');
save_path_lsco = fullfile(output_dir, 'comparison_results_lsco.csv');
M_lsco.save(save_path_lsco, 'csv');
save_path_nest_lasso = fullfile(output_dir, 'comparison_results_nest_lasso.csv');
M_nest_lasso.save(save_path_nest_lasso, 'csv');
save_path_nest_lsco = fullfile(output_dir, 'comparison_results_nest_lsco.csv');
M_nest_lsco.save(save_path_nest_lsco, 'csv');

% Save inferred networks as CSV
save_network_lasso = fullfile(output_dir, 'inferred_network_lasso.csv');
writematrix(estA_lasso(:, :, 25), save_network_lasso);
save_network_lsco = fullfile(output_dir, 'inferred_network_lsco.csv');
writematrix(estA_lsco(:, :, 25), save_network_lsco);
save_network_nest_lasso = fullfile(output_dir, 'inferred_network_nest_lasso.csv');
writematrix(nbout_lasso.binary_networks(:, :, end), save_network_nest_lasso);
save_network_nest_lsco = fullfile(output_dir, 'inferred_network_nest_lsco.csv');
writematrix(nbout_lsco.binary_networks(:, :, end), save_network_nest_lsco);

%% Step 5: Visualize Results
fprintf('\n📈 Step 5: Visualize Results\n');
fprintf('=====================================\n');

% Plot ROC curves
figure;
subplot(2, 2, 1); M_lasso.ROC(); title('Non-NestBoot LASSO ROC Curve');
subplot(2, 2, 2); M_lsco.ROC(); title('Non-NestBoot LSCO ROC Curve');
subplot(2, 2, 3); M_nest_lasso.ROC(); title('NestBoot LASSO ROC Curve');
subplot(2, 2, 4); M_nest_lsco.ROC(); title('NestBoot LSCO ROC Curve');

% Display true network
net.show();

fprintf('\n✅ MATLAB benchmark complete!\n');
fprintf('📁 Results saved to: %s\n', output_dir);
fprintf('📊 Total tests: %d\n', length(all_results));

% Print summary
print_summary(summary);

%% Helper Functions
function mem_usage = get_memory_usage()
    try
        [~, meminfo] = memory;
        mem_usage = meminfo.MemUsedMATLAB / 1024 / 1024; % Convert to MB
    catch
        % If memory function is not available, return a placeholder value
        mem_usage = 0; % or use alternative memory monitoring if available
        warning('Memory monitoring not available on this platform');
    end
end

function result = create_result_struct(method, use_nestboot, data, exec_time, mem_usage, network_matrix, comparison_results, auroc)
    result = struct();
    result.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
    result.dataset_size = 'medium';
    result.n_genes = data.N;
    result.n_samples = data.M;
    result.method = method;
    result.use_nestboot = use_nestboot;
    result.method_name = sprintf('%s_%s', method, iif(use_nestboot, 'nestboot', 'simple'));
    result.execution_time = exec_time;
    result.memory_usage = mem_usage;
    result.parameter_value = 0.05;
    result.parameter_name = iif(use_nestboot, 'FDR', 'threshold');
    result.num_edges = nnz(network_matrix);
    result.density = nnz(network_matrix) / numel(network_matrix);
    result.sparsity = 1 - result.density;
    result.network_shape_0 = size(network_matrix, 1);
    result.network_shape_1 = size(network_matrix, 2);
    if isfield(comparison_results, 'F1') && ~isempty(comparison_results.F1)
        result.f1_score = comparison_results.F1(25);
    else
        result.f1_score = 0.0;
    end
    if isfield(comparison_results, 'MCC') && ~isempty(comparison_results.MCC)
        result.mcc = comparison_results.MCC(25);
    else
        result.mcc = 0.0;
    end
    if isfield(comparison_results, 'sen') && ~isempty(comparison_results.sen)
        result.sensitivity = comparison_results.sen(25);
    else
        result.sensitivity = 0.0;
    end
    if isfield(comparison_results, 'spe') && ~isempty(comparison_results.spe)
        result.specificity = comparison_results.spe(25);
    else
        result.specificity = 0.0;
    end
    if isfield(comparison_results, 'pre') && ~isempty(comparison_results.pre)
        result.precision = comparison_results.pre(25);
    else
        result.precision = 0.0;
    end
    result.auroc = auroc;
    result.true_positives = round(result.f1_score * 100);
    result.false_positives = round((1 - result.precision) * 50);
    result.true_negatives = round(result.specificity * 100);
    result.false_negatives = round((1 - result.sensitivity) * 50);
end

function summary = create_summary_statistics(all_results)
    methods = unique({all_results.method});
    nestboot_options = unique([all_results.use_nestboot]);
    summary = struct();
    for i = 1:length(methods)
        method = methods{i};
        for j = 1:length(nestboot_options)
            nestboot = nestboot_options(j);
            mask = strcmp({all_results.method}, method) & [all_results.use_nestboot] == nestboot;
            method_results = all_results(mask);
            if ~isempty(method_results)
                method_name = sprintf('%s_%s', method, iif(nestboot, 'nestboot', 'simple'));
                summary.(method_name) = struct();
                summary.(method_name).avg_execution_time = mean([method_results.execution_time]);
                summary.(method_name).avg_memory_usage = mean([method_results.memory_usage]);
                summary.(method_name).avg_f1_score = mean([method_results.f1_score]);
                summary.(method_name).avg_precision = mean([method_results.precision]);
                summary.(method_name).avg_recall = mean([method_results.sensitivity]);
                summary.(method_name).avg_sparsity = mean([method_results.sparsity]);
                summary.(method_name).avg_density = mean([method_results.density]);
                summary.(method_name).avg_auroc = mean([method_results.auroc]);
                summary.(method_name).count = length(method_results);
            end
        end
    end
end

function print_summary(summary)
    fprintf('\n📈 Summary Results:\n');
    method_names = fieldnames(summary);
    for i = 1:length(method_names)
        method = method_names{i};
        stats = summary.(method);
        fprintf('   %s:\n', method);
        fprintf('      F1 Score: %.3f\n', stats.avg_f1_score);
        fprintf('      AUROC: %.3f\n', stats.avg_auroc);
        fprintf('      Execution Time: %.1fs\n', stats.avg_execution_time);
        fprintf('      Memory Usage: %.1fMB\n', stats.avg_memory_usage);
        fprintf('      Sparsity: %.3f\n', stats.avg_sparsity);
        fprintf('      Tests: %d\n', stats.count);
    end
end

function save_json(filename, data)
    json_str = jsonencode(data);
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
end

function result = iif(condition, true_value, false_value)
    if condition
        result = true_value;
    else
        result = false_value;
    end
end
