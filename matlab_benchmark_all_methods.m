% MATLAB All-Methods N50 Benchmark
% Runs all 5 methods (LSCO, LASSO, CLR, GENIE3, TIGRESS) on all N50 datasets
% Date: 2025-11-25

%% Setup
clear all;
rng(42); % Set random seed
output_dir = 'benchmark_results';
figures_dir = 'benchmark_figures';

if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if ~exist(figures_dir, 'dir'), mkdir(figures_dir); end

addpath('../Volume1/dmorgan/genespider/'); % Add GeneSPIDER2 library

% Initialize performance tracking
all_results = [];
fprintf('🚀 MATLAB All-Methods N50 Benchmark\n');
fprintf('========================================\n');

%% Configuration
% Dataset and network paths
dataset_root = '~/Downloads/gs-datasets/N50';
network_root = '~/Downloads/gs-networks';

% Zetavec for LASSO and LSCO
zetavec = logspace(-6, 0, 30); % Sparsity parameters
best_idx = 25; % Index to use for results (MATLAB standard)

%% Find all N50 datasets
dataset_files = dir(fullfile(dataset_root, '*.json'));
fprintf('Found %d N50 datasets.\n', length(dataset_files));

%% Process all datasets
for i = 1:length(dataset_files)
    dataset_filename = dataset_files(i).name;
    dataset_path = fullfile(dataset_root, dataset_filename);
    
    fprintf('\n🔄 [%d/%d] Processing %s\n', i, length(dataset_files), dataset_filename);
    
    try
        % Load dataset
        data = datastruct.Dataset.fetch(['file://' dataset_path]);
        
        % Extract network ID from dataset
        network_id = [];
        if isfield(data, 'network') && ~isempty(data.network)
            % Extract ID from network field
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
        
        % Load network
        net = datastruct.Network.fetch(['file://' network_file]);
        
        fprintf('   📂 Network: %s\n', network_file);
        
        % Run all methods
        methods = {'LASSO', 'LSCO', 'CLR', 'GENIE3', 'TIGRESS'};
        
        for m = 1:length(methods)
            method_name = methods{m};
            fprintf('   Running %s...\n', method_name);
            
            [estA, exec_time, mem_usage, error_msg] = run_method(method_name, data, net, zetavec);
            
            if ~isempty(estA)
                % Compare with true network
                result = compare_and_report(method_name, net, estA, best_idx, exec_time, mem_usage, data, dataset_filename, network_file);
                all_results = [all_results; result];
                
                fprintf('      ✅ F1: %.3f, AUROC: %.3f, Time: %.1fs\n', ...
                    result.f1_score, result.auroc, exec_time);
            else
                fprintf('   ❌ %s failed: %s\n', method_name, error_msg);
            end
        end
        
        % Save results after each dataset
        results_table = struct2table(all_results);
        writetable(results_table, fullfile(output_dir, 'matlab_all_methods_results.csv'));
        
    catch ME
        fprintf('   ❌ Error processing %s: %s\n', dataset_filename, ME.message);
    end
end

%% Save final results
fprintf('\n💾 Saving Final Results\n');
fprintf('=====================================\n');

% Save benchmark results as CSV
results_table = struct2table(all_results);
writetable(results_table, fullfile(output_dir, 'matlab_all_methods_results.csv'));

% Save summary statistics
summary = create_summary_statistics(all_results);
save_json(fullfile(output_dir, 'matlab_all_methods_summary.json'), summary);

fprintf('✅ Results saved to %s\n', output_dir);

%% Generate Summary
fprintf('\n📊 Summary by Method:\n');
method_names = unique({all_results.method});
for i = 1:length(method_names)
    method = method_names{i};
    mask = strcmp({all_results.method}, method);
    method_results = all_results(mask);
    if ~isempty(method_results)
        fprintf('   %s:\n', method);
        fprintf('      F1: %.3f ± %.3f\n', mean([method_results.f1_score]), std([method_results.f1_score]));
        fprintf('      AUROC: %.3f ± %.3f\n', mean([method_results.auroc]), std([method_results.auroc]));
        fprintf('      Time: %.1f ± %.1fs\n', mean([method_results.execution_time]), std([method_results.execution_time]));
        fprintf('      Tests: %d\n', length(method_results));
    end
end

fprintf('\n✅ MATLAB all-methods benchmark complete!\n');
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

function [estA, exec_time, mem_usage, error_msg] = run_method(method_name, data, net, zetavec)
    % Run a specific inference method
    tic;
    mem_start = get_memory_usage();
    error_msg = '';
    
    try
        if strcmp(method_name, 'LASSO')
            % LASSO: Methods.lasso(data, net, zetavec, false)
            estA = Methods.lasso(data, net, zetavec, false);
        elseif strcmp(method_name, 'LSCO')
            % LSCO: Methods.lsco(data, net, zetavec, false, 'input')
            estA = Methods.lsco(data, net, zetavec, false, 'input');
        elseif strcmp(method_name, 'CLR')
            % CLR: Try different possible signatures
            try
                estA = Methods.CLR(data, zetavec);
            catch
                try
                    estA = Methods.CLR(data);
                catch
                    estA = Methods.CLR(data, net, zetavec);
                end
            end
        elseif strcmp(method_name, 'GENIE3')
            % GENIE3: Try different possible signatures
            try
                estA = Methods.GENIE3(data, zetavec);
            catch
                try
                    estA = Methods.GENIE3(data);
                catch
                    estA = Methods.GENIE3(data, net, zetavec);
                end
            end
        elseif strcmp(method_name, 'TIGRESS')
            % TIGRESS: Try different possible signatures
            try
                estA = Methods.TIGRESS(data, zetavec);
            catch
                try
                    estA = Methods.TIGRESS(data);
                catch
                    estA = Methods.TIGRESS(data, net, zetavec);
                end
            end
        else
            error('Unknown method: %s', method_name);
        end
    catch ME
        fprintf('   ⚠️  Error running %s: %s\n', method_name, ME.message);
        error_msg = ME.message;
        estA = [];
        exec_time = toc;
        mem_usage = 0;
        return;
    end
    
    exec_time = toc;
    mem_usage = get_memory_usage() - mem_start;
end

function result = compare_and_report(method_name, net, estA, idx, exec_time, mem_usage, data, dataset_filename, network_filename)
    % Compare inferred network with true network
    M = analyse.CompareModels(net, estA);
    results_struct = struct(M);
    
    result = create_result_struct(lower(method_name), false, data, exec_time, mem_usage, ...
        estA(:, :, idx), results_struct, M.AUROC(), idx);
    result.dataset = dataset_filename;
    result.network = network_filename;
    result.timestamp = datestr(now, 'yyyy-mm-ddTHH:MM:SS');
end

function mem_usage = get_memory_usage()
    try
        [~, meminfo] = memory;
        mem_usage = meminfo.MemUsedMATLAB / 1024 / 1024; % Convert to MB
    catch
        mem_usage = 0; % Memory monitoring not available
    end
end

function result = create_result_struct(method, use_nestboot, data, exec_time, mem_usage, network_matrix, comparison_results, auroc, idx)
    result = struct();
    result.method = method;
    result.use_nestboot = use_nestboot;
    result.execution_time = exec_time;
    result.memory_usage = mem_usage;
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
end

function summary = create_summary_statistics(all_results)
    methods = unique({all_results.method});
    summary = struct();
    
    for i = 1:length(methods)
        method = methods{i};
        mask = strcmp({all_results.method}, method);
        method_results = all_results(mask);
        
        if ~isempty(method_results)
            method_name = method;
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

function save_json(filename, data)
    json_str = jsonencode(data);
    fid = fopen(filename, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
end