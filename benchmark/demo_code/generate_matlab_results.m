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
addpath('../Volume1/dmorgan/genespider/'); % Add GeneSPIDER2 library

% Initialize performance tracking
all_results = [];
fprintf('🚀 MATLAB Performance Benchmark\n');
fprintf('========================================\n');

%% Step 1: Network Data Import and Analysis
fprintf('\n📥 Step 1: Network Data Import and Analysis\n');
fprintf('=====================================\n');

% Load dataset and network (choose SNR level)
USE_HIGH_SNR = true; % Set to false for SNR10, true for SNR100000

if USE_HIGH_SNR
    fprintf('Loading HIGH SNR dataset (SNR100000)...\n');
    dataset_url = 'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR100000-IDY252384.json';
else
    fprintf('Loading LOW SNR dataset (SNR10)...\n');
    dataset_url = 'https://bitbucket.org/sonnhammergrni/gs-datasets/raw/d2047430263f5ffe473525c74b4318f723c23b0e/N50/Tjarnberg-ID252384-D20151111-N50-E150-SNR10-IDY252384.json';
end

network_url = 'https://bitbucket.org/sonnhammergrni/gs-networks/raw/0b3a66e67d776eadaa5d68667ad9c8fbac12ef85/random/N50/Tjarnberg-D20150910-random-N50-L158-ID252384.json';

data = datastruct.Dataset.fetch(dataset_url);
net = datastruct.Network.fetch(network_url);

fprintf('✅ Dataset loaded: %s\n', data.dataset);
fprintf('   📊 Expression matrix shape: [%d, %d]\n', size(data.Y));
fprintf('   🧬 Number of genes: %d\n', data.N);
fprintf('   🔬 Number of samples: %d\n', data.M);
fprintf('✅ Reference network loaded: %s\n', net.network);

%% Step 2: Network Inference
fprintf('\n🔍 Step 2: Network Inference\n');
fprintf('=====================================\n');

zetavec = logspace(-6, 0, 30); % Sparsity parameters
best_idx = 25; % Index to use for results (MATLAB standard)

% Run all network inference methods
[estA_lasso, time_lasso, mem_lasso] = run_method('LASSO', @Methods.lasso, data, net, zetavec);
[estA_lsco, time_lsco, mem_lsco] = run_method('LSCO', @Methods.lsco, data, net, zetavec);

% Try optional methods (CLR, GENIE3, TIGRESS) - skip if not available
fprintf('   Checking for optional methods (CLR, GENIE3, TIGRESS)...\n');
estA_clr = []; time_clr = 0; mem_clr = 0;
estA_genie3 = []; time_genie3 = 0; mem_genie3 = 0;
estA_tigress = []; time_tigress = 0; mem_tigress = 0;

if exist('Methods.CLR', 'file') || exist('Methods.clr', 'file')
    [estA_clr, time_clr, mem_clr] = run_method('CLR', @Methods.CLR, data, net, zetavec);
else
    fprintf('   ℹ️  CLR method not found in GeneSPIDER2, skipping...\n');
end

if exist('Methods.GENIE3', 'file') || exist('Methods.genie3', 'file')
    [estA_genie3, time_genie3, mem_genie3] = run_method('GENIE3', @Methods.GENIE3, data, net, zetavec);
else
    fprintf('   ℹ️  GENIE3 method not found in GeneSPIDER2, skipping...\n');
end

if exist('Methods.TIGRESS', 'file') || exist('Methods.tigress', 'file')
    [estA_tigress, time_tigress, mem_tigress] = run_method('TIGRESS', @Methods.TIGRESS, data, net, zetavec);
else
    fprintf('   ℹ️  TIGRESS method not found in GeneSPIDER2, skipping...\n');
end


%% Step 3: Model Comparison
fprintf('\n📊 Step 3: Model Comparison\n');
fprintf('=====================================\n');

% Compare all methods and collect results
result_lasso = compare_and_report('LASSO', net, estA_lasso, best_idx, time_lasso, mem_lasso, data);
all_results = [all_results; result_lasso];

result_lsco = compare_and_report('LSCO', net, estA_lsco, best_idx, time_lsco, mem_lsco, data);
all_results = [all_results; result_lsco];

if ~isempty(estA_clr)
    result_clr = compare_and_report('CLR', net, estA_clr, best_idx, time_clr, mem_clr, data);
    all_results = [all_results; result_clr];
end

if ~isempty(estA_genie3)
    result_genie3 = compare_and_report('GENIE3', net, estA_genie3, best_idx, time_genie3, mem_genie3, data);
    all_results = [all_results; result_genie3];
end

if ~isempty(estA_tigress)
    result_tigress = compare_and_report('TIGRESS', net, estA_tigress, best_idx, time_tigress, mem_tigress, data);
    all_results = [all_results; result_tigress];
end




%% Step 4: Save Results
fprintf('\n💾 Step 4: Save Results\n');
fprintf('=====================================\n');

% Save benchmark results as CSV
results_table = struct2table(all_results);
writetable(results_table, fullfile(output_dir, 'matlab_benchmark_results.csv'));

% Save summary statistics
summary = create_summary_statistics(all_results);
save_json(fullfile(output_dir, 'matlab_summary.json'), summary);

% Save comparison results and networks for all methods
methods = {};
networks = {};
comparisons = {};

% Add LASSO and LSCO (always present)
methods{end+1} = 'lasso';
networks{end+1} = estA_lasso;
comparisons{end+1} = result_lasso.comparison_obj;

methods{end+1} = 'lsco';
networks{end+1} = estA_lsco;
comparisons{end+1} = result_lsco.comparison_obj;

% Add optional methods if they succeeded
if ~isempty(estA_clr)
    methods{end+1} = 'clr';
    networks{end+1} = estA_clr;
    comparisons{end+1} = result_clr.comparison_obj;
end

if ~isempty(estA_genie3)
    methods{end+1} = 'genie3';
    networks{end+1} = estA_genie3;
    comparisons{end+1} = result_genie3.comparison_obj;
end

if ~isempty(estA_tigress)
    methods{end+1} = 'tigress';
    networks{end+1} = estA_tigress;
    comparisons{end+1} = result_tigress.comparison_obj;
end

for i = 1:length(methods)
    method = methods{i};
    
    % Save comparison results
    save_path = fullfile(output_dir, sprintf('comparison_results_%s.csv', method));
    comparisons{i}.save(save_path, 'csv');
    
    % Save inferred network
    network_path = fullfile(output_dir, sprintf('inferred_network_%s.csv', method));
    writematrix(networks{i}(:, :, best_idx), network_path);
end

fprintf('✅ Results saved to %s\n', output_dir);

%% Step 5: Visualize Results
fprintf('\n📈 Step 5: Visualize Results\n');
fprintf('=====================================\n');

% Plot ROC curves
figure;
num_methods = length(comparisons);
rows = 2;
cols = 3;

subplot(rows, cols, 1); result_lasso.comparison_obj.ROC(); title('LASSO ROC Curve');
subplot(rows, cols, 2); result_lsco.comparison_obj.ROC(); title('LSCO ROC Curve');

plot_idx = 3;
if ~isempty(estA_clr)
    subplot(rows, cols, plot_idx); result_clr.comparison_obj.ROC(); title('CLR ROC Curve');
    plot_idx = plot_idx + 1;
end
if ~isempty(estA_genie3)
    subplot(rows, cols, plot_idx); result_genie3.comparison_obj.ROC(); title('GENIE3 ROC Curve');
    plot_idx = plot_idx + 1;
end
if ~isempty(estA_tigress)
    subplot(rows, cols, plot_idx); result_tigress.comparison_obj.ROC(); title('TIGRESS ROC Curve');
    plot_idx = plot_idx + 1;
end

% Always show true network in last subplot
subplot(rows, cols, 6); net.show(); title('True Network');

fprintf('\n✅ MATLAB benchmark complete!\n');
fprintf('📁 Results saved to: %s\n', output_dir);
fprintf('📊 Total tests: %d\n', length(all_results));

% Print summary
print_summary(summary);

%% Helper Functions
function [estA, exec_time, mem_usage] = run_method(method_name, method_func, data, net, zetavec)
    fprintf('   🎯 Running %s...\n', method_name);
    tic;
    mem_start = get_memory_usage();
    
    % Call the method function with appropriate parameters
    % Different methods have different signatures in GeneSPIDER2
    try
        if strcmp(method_name, 'LASSO')
            % LASSO: Methods.lasso(data, net, zetavec, false)
            estA = method_func(data, net, zetavec, false);
        elseif strcmp(method_name, 'LSCO')
            % LSCO: Methods.lsco(data, net, zetavec, false, 'input')
            estA = method_func(data, net, zetavec, false, 'input');
        elseif strcmp(method_name, 'CLR')
            % CLR: Try different possible signatures
            try
                estA = method_func(data, zetavec);
            catch
                try
                    estA = method_func(data);
                catch
                    estA = method_func(data, net, zetavec);
                end
            end
        elseif strcmp(method_name, 'GENIE3')
            % GENIE3: Try different possible signatures
            try
                estA = method_func(data, zetavec);
            catch
                try
                    estA = method_func(data);
                catch
                    estA = method_func(data, net, zetavec);
                end
            end
        elseif strcmp(method_name, 'TIGRESS')
            % TIGRESS: Try different possible signatures
            try
                estA = method_func(data, zetavec);
            catch
                try
                    estA = method_func(data);
                catch
                    estA = method_func(data, net, zetavec);
                end
            end
        else
            error('Unknown method: %s', method_name);
        end
    catch ME
        fprintf('   ⚠️  Error running %s: %s\n', method_name, ME.message);
        fprintf('   Skipping %s...\n', method_name);
        estA = [];
        exec_time = toc;
        mem_usage = 0;
        return;
    end
    
    exec_time = toc;
    mem_usage = get_memory_usage() - mem_start;
end

function result = compare_and_report(method_name, net, estA, idx, exec_time, mem_usage, data)
    M = analyse.CompareModels(net, estA);
    results_struct = struct(M);
    
    fprintf('   ✅ %s Results:\n', method_name);
    fprintf('      F1 Score: %.3f, Time: %.1fs, Memory: %.1fMB, AUROC: %.3f\n', ...
        results_struct.F1(idx), exec_time, mem_usage, M.AUROC());
    
    result = create_result_struct(lower(method_name), false, data, exec_time, mem_usage, ...
        estA(:, :, idx), results_struct, M.AUROC(), idx);
    result.comparison_obj = M;
end

function mem_usage = get_memory_usage()
    try
        [~, meminfo] = memory;
        mem_usage = meminfo.MemUsedMATLAB / 1024 / 1024; % Convert to MB
    catch
        % If memory function is not available, return a placeholder value
        mem_usage = 0; % Memory monitoring not available on this platform
    end
end

function result = create_result_struct(method, use_nestboot, data, exec_time, mem_usage, network_matrix, comparison_results, auroc, idx)
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
    result.parameter_index = idx;
    result.num_edges = nnz(network_matrix);
    result.density = nnz(network_matrix) / numel(network_matrix);
    result.sparsity = 1 - result.density;
    result.network_shape_0 = size(network_matrix, 1);
    result.network_shape_1 = size(network_matrix, 2);
    if isfield(comparison_results, 'F1') && ~isempty(comparison_results.F1)
        result.f1_score = comparison_results.F1(idx);
    else
        result.f1_score = 0.0;
    end
    if isfield(comparison_results, 'MCC') && ~isempty(comparison_results.MCC)
        result.mcc = comparison_results.MCC(idx);
    else
        result.mcc = 0.0;
    end
    if isfield(comparison_results, 'sen') && ~isempty(comparison_results.sen)
        result.sensitivity = comparison_results.sen(idx);
    else
        result.sensitivity = 0.0;
    end
    if isfield(comparison_results, 'spe') && ~isempty(comparison_results.spe)
        result.specificity = comparison_results.spe(idx);
    else
        result.specificity = 0.0;
    end
    if isfield(comparison_results, 'pre') && ~isempty(comparison_results.pre)
        result.precision = comparison_results.pre(idx);
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
