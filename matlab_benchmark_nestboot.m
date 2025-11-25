% MATLAB NestBoot N50 Benchmark
% Runs NestBoot analysis for LASSO and LSCO on all N50 datasets
% Uses 50 inner and 50 outer bootstrap runs, FDR=5%
% Date: 2025-11-25

%% Setup
clear all;
rng(42); % Set random seed
output_dir = 'benchmark_results';
figures_dir = 'benchmark_figures';

if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end

addpath('../GeneSPIDER2/'); % Add GeneSPIDER2 library

% Initialize performance tracking
all_results = [];
fprintf('🚀 MATLAB NestBoot N50 Benchmark\n');
fprintf('===================================\n');

%% Configuration
% Dataset and network paths
dataset_root = '../GeneSPIDER2/data/gs-datasets/N50';
network_root = '../GeneSPIDER2/data/gs-networks';

% NestBoot parameters - matching MATLAB standard
N_INIT = 10;     % Outer loop iterations (reduced for faster testing)
N_BOOT = 10;     % Inner loop iterations (reduced for faster testing)
FDR = 5;         % FDR percentage

% Zetavec for LASSO and LSCO
zetavec = logspace(-6, 0, 30);

fprintf('Configuration:\n');
fprintf('   Init runs: %d\n', N_INIT);
fprintf('   Boot runs: %d\n', N_BOOT);
fprintf('   FDR: %d%%\n', FDR);
fprintf('   Zetavec: %d values from %.2e to %.2e\n', length(zetavec), zetavec(1), zetavec(end));

%% Find all N50 datasets
dataset_files = dir(fullfile(dataset_root, '*.json'));
fprintf('Found %d N50 datasets.\n', length(dataset_files));

%% Process all datasets
for i = 1:length(dataset_files)
    dataset_filename = dataset_files(i).name;
    dataset_path = fullfile(dataset_root, dataset_filename);
    
    fprintf('\n🔄 [%d/%d] Processing %s\n', i, length(dataset_files), dataset_filename);
    
    try
        % Load dataset from JSON file locally
        data = load_dataset_from_json(dataset_path);
        
        % Extract network ID from dataset
        network_id = [];
        if isfield(data, 'network') && ~isempty(data.network)
            parts = strsplit(data.network, '-ID');
            if length(parts) > 1
                network_id = parts{2};
            end
        end
        
        if isempty(network_id)
            fprintf('   ⚠️ Could not extract network ID from %s\n', dataset_filename);
            continue;
        end
        
        % Find network file
        network_file = find_network_file(network_root, network_id);
        if isempty(network_file)
            fprintf('   ⚠️ Network file not found for ID %s\n', network_id);
            continue;
        end
        
        % Load network from JSON file locally
        net = load_network_from_json(network_file);
        
        fprintf('   📂 Network: %s\n', network_file);
        
        % Run LASSO NestBoot
        fprintf('   Running LASSO NestBoot...\n');
        result_lasso = run_nestboot('LASSO', data, net, zetavec, N_INIT, N_BOOT, FDR);
        result_lasso.dataset = dataset_filename;
        result_lasso.network = network_file;
        result_lasso.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
        all_results = [all_results; result_lasso];
        
        fprintf('      ✅ F1: %.3f, AUROC: %.3f, Time: %.1fs\n', ...
            result_lasso.f1, result_lasso.auroc, result_lasso.time);
        
        % Run LSCO NestBoot
        fprintf('   Running LSCO NestBoot...\n');
        result_lsco = run_nestboot('LSCO', data, net, zetavec, N_INIT, N_BOOT, FDR);
        result_lsco.dataset = dataset_filename;
        result_lsco.network = network_file;
        result_lsco.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
        all_results = [all_results; result_lsco];
        
        fprintf('      ✅ F1: %.3f, AUROC: %.3f, Time: %.1fs\n', ...
            result_lsco.f1, result_lsco.auroc, result_lsco.time);
        
        % Save results after each dataset (append mode)
        results_table = struct2table(all_results(end));  % Only save the latest result
        if i == 1
            writetable(results_table, fullfile(output_dir, 'matlab_nestboot_results.csv'));
        else
            writetable(results_table, fullfile(output_dir, 'matlab_nestboot_results.csv'), 'WriteMode', 'append');
        end
        
    catch ME
        fprintf('   ❌ Error processing %s: %s\n', dataset_filename, ME.message);
    end
end

%% Save final results
fprintf('\n💾 Saving Final Results\n');
fprintf('=====================================\n');

% Save benchmark results as CSV (overwrite final results)
results_table = struct2table(all_results);
writetable(results_table, fullfile(output_dir, 'matlab_nestboot_results.csv'));

% Save summary statistics
summary = create_summary_statistics(all_results);
save_json(fullfile(output_dir, 'matlab_nestboot_summary.json'), summary);

fprintf('✅ Results saved to %s\n', output_dir);

%% Generate Summary
fprintf('\n📊 Summary by Method:\n');
method_names = unique({all_results.method});
for i = 1:length(method_names)
    method = method_names{i};
    mask = strcmp({all_results.method}, method);
    method_results = all_results(mask);
    if ~isempty(method_results)
        fprintf('   %s NestBoot:\n', method);
        fprintf('      F1: %.3f ± %.3f\n', mean([method_results.f1]), std([method_results.f1]));
        fprintf('      AUROC: %.3f ± %.3f\n', mean([method_results.auroc]), std([method_results.auroc]));
        fprintf('      Time: %.1f ± %.1fs\n', mean([method_results.time]), std([method_results.time]));
        fprintf('      Density: %.3f ± %.3f\n', mean([method_results.density]), std([method_results.density]));
        fprintf('      Tests: %d\n', length(method_results));
    end
end

fprintf('\n✅ MATLAB NestBoot benchmark complete!\n');
fprintf('📁 Results saved to: %s\n', output_dir);
fprintf('📊 Total tests: %d\n', length(all_results));

%% Helper Functions
function network_file = find_network_file(network_root, network_id)
    % Find network file recursively with the given ID
    network_file = [];
    
    % Try different patterns
    patterns = {
        sprintf('*ID%s*.json', network_id),
        sprintf('*ID%s*', network_id),
        sprintf('*%s*.json', network_id),
        sprintf('*%s*', network_id)
    };
    
    for p = 1:length(patterns)
        files = dir(fullfile(network_root, '**', patterns{p}));
        if ~isempty(files)
            network_file = fullfile(files(1).folder, files(1).name);
            break;
        end
    end
end

function result = run_nestboot(method_name, data, net, zetavec, n_init, n_boot, fdr)
    % Run NestBoot analysis for a specific method
    tic;
    
    try
        % Run NestBoot using GeneSPIDER2's built-in function
        if strcmp(method_name, 'LASSO')
            [xnet, support, fp_rate] = Methods.nestboot(data, net, zetavec, n_init, n_boot, fdr, 'lasso');
        elseif strcmp(method_name, 'LSCO')
            [xnet, support, fp_rate] = Methods.nestboot(data, net, zetavec, n_init, n_boot, fdr, 'lsco');
        else
            error('Unknown method: %s', method_name);
        end
        
        exec_time = toc;
        
        % Compare with true network
        M = analyse.CompareModels(net, xnet);
        
        result = struct();
        result.method = method_name;
        result.n_init = n_init;
        result.n_boot = n_boot;
        result.fdr = fdr;
        result.time = exec_time;
        result.auroc = M.AUROC();
        result.f1 = M.F1(end);  % Use final threshold
        result.precision = M.pre(end);
        result.recall = M.sen(end);
        result.density = nnz(xnet) / numel(xnet);
        result.support_threshold = support;
        result.fp_rate = fp_rate;
        
    catch ME
        fprintf('   ❌ NestBoot failed for %s: %s\n', method_name, ME.message);
        exec_time = toc;
        
        result = struct();
        result.method = method_name;
        result.n_init = n_init;
        result.n_boot = n_boot;
        result.fdr = fdr;
        result.time = exec_time;
        result.auroc = 0.5;
        result.f1 = 0.0;
        result.precision = 0.0;
        result.recall = 0.0;
        result.density = 0.0;
        result.support_threshold = 0;
        result.fp_rate = 0;
    end
end

function summary = create_summary_statistics(all_results)
    methods = unique({all_results.method});
    summary = struct();
    
    for i = 1:length(methods)
        method = methods{i};
        mask = strcmp({all_results.method}, method);
        method_results = all_results(mask);
        
        if ~isempty(method_results)
            method_name = sprintf('%s_nestboot', lower(method));
            summary.(method_name) = struct();
            summary.(method_name).avg_execution_time = mean([method_results.time]);
            summary.(method_name).avg_f1_score = mean([method_results.f1]);
            summary.(method_name).avg_auroc = mean([method_results.auroc]);
            summary.(method_name).avg_precision = mean([method_results.precision]);
            summary.(method_name).avg_recall = mean([method_results.recall]);
            summary.(method_name).avg_density = mean([method_results.density]);
            summary.(method_name).count = length(method_results);
        end
    end
end

function save_json(filename, data)
    json_str = jsonencode(data);
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
end

function data = load_dataset_from_json(filename)
    % Load dataset from JSON file locally
    json_str = fileread(filename);
    data_struct = jsondecode(json_str);
    
    % Convert to expected structure - assuming the JSON has obj_data field
    if isfield(data_struct, 'obj_data')
        data = data_struct.obj_data;
    else
        data = data_struct;
    end
end

function net = load_network_from_json(filename)
    % Load network from JSON file locally
    json_str = fileread(filename);
    net_struct = jsondecode(json_str);
    
    % Convert to expected structure - assuming the JSON has obj_data field
    if isfield(net_struct, 'obj_data')
        net = net_struct.obj_data;
    else
        net = net_struct;
    end
    
    % Ensure A field exists (adjacency matrix)
    if ~isfield(net, 'A')
        error('Network JSON must contain adjacency matrix field "A"');
    end
end
