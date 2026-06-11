# **Gene Regulatory Network Simulations and Benchmarking with GeneSPIDER2**

## **Abstract**

Gene regulatory networks (GRNs) represent the complex interactions between genes and their regulatory elements, forming the foundation of cellular function and behavior. Accurate inference of GRNs from gene expression data is a central challenge in systems biology, requiring robust computational methods and rigorous benchmarking approaches. GeneSPIDER2 (Generation and Simulation Package for Informative Data ExploRation, version 2\) is a comprehensive MATLAB toolbox designed for GRN simulation, inference, and benchmarking with controlled network and data properties. This protocol provides a detailed guide for using GeneSPIDER2 to generate synthetic gene regulatory networks with realistic topological properties, simulate both bulk and single-cell gene expression data under various perturbation conditions, infer GRNs using multiple computational methods, and benchmark inference performance against ground truth networks. The protocol covers the theoretical foundations of GRN modeling, including the mathematical framework for network stability and data simulation, practical implementation steps for working with both synthetic and experimental data, and advanced applications such as nested bootstrap analysis for controlling false discovery rates. This comprehensive approach enables researchers to evaluate GRN inference methods under controlled conditions, understand the impact of network and data properties on inference accuracy, and apply these methods to real biological datasets. The entire workflow can be completed in approximately 2-4 hours for small networks (50-100 genes) and scales to large-scale networks (thousands of genes) with appropriate computational resources.

1. ## **Introduction**

### 

   1. ### **Background and Rationale**

Gene regulatory networks are fundamental to understanding how cells coordinate gene expression in response to developmental cues, environmental stimuli, and disease states. The inference of GRNs from high-throughput gene expression data has been an active area of research for over two decades, yielding numerous computational methods that exploit different mathematical and statistical frameworks (1). However, evaluating the accuracy and reliability of these methods remains challenging due to the lack of complete ground truth networks in biological systems and the difficulty in controlling experimental conditions.

The advent of single-cell RNA sequencing (scRNA-seq) technologies has revolutionized our ability to measure gene expression at unprecedented resolution, enabling the inference of cell-type-specific regulatory networks and the study of cellular heterogeneity (2). Recent developments in CRISPR-based perturbation screens, such as Perturb-seq and CROP-seq, have further enhanced our capacity to probe gene function by combining genetic perturbations with single-cell transcriptomics (3, 4). These experimental advances necessitate computational tools that can simulate realistic perturbed single-cell data for benchmarking inference methods.

GeneSPIDER2 addresses these needs by providing a unified framework for generating synthetic GRNs with biologically realistic properties, simulating gene expression data under various experimental designs, and benchmarking inference performance. The toolbox builds upon the original GeneSPIDER (5) with major improvements including: (i) the ability to generate large-scale GRNs (up to 20,000 genes) with scale-free degree distributions and modular structure, (ii) simulation of perturbed single-cell RNA-seq data with realistic noise characteristics, and (iii) enhanced computational efficiency and stability (6).

2. ### **Theoretical Framework**

#### **Gene Regulatory Network Representation**

In GeneSPIDER2, a GRN is represented as a directed, weighted, and signed adjacency matrix *A* ∈ ℝ*g*×*g*, where *g* is the number of genes. Each element *aij* represents the regulatory effect of gene *j* on gene *i*, where positive values indicate activation, negative values indicate repression, and zero indicates no direct interaction. It is critical to note that in GeneSPIDER2, the direction of regulation is from column to row, which differs from some other network representations.

The steady-state relationship between gene perturbations and the resulting gene expression changes can be described by a linear time-invariant system:

[![][image1]](https://www.codecogs.com/eqnedit.php?latex=%20X%20%3D%20-A%5E%7B-1%7DP%20#0)  
where *X* ∈ ℝ*g*×*e* is the noise-free gene expression matrix representing logarithmic fold-changes relative to control, *P* ∈ ℝ*g*×*e* is the perturbation design matrix defining which genes are perturbed in each of *e* experiments or cells, and *A*\-1 is often referred to as the static gain matrix *G* (6). This formulation assumes that the network reaches a new steady state following each perturbation, which is reasonable for knockdown experiments with sufficient time for the system to equilibrate.

#### **Network Stability**

For a GRN to be biologically meaningful and mathematically tractable, it must be stable. Stability ensures that small perturbations do not lead to unbounded gene expression changes. A linear system represented by matrix *A* is stable if all eigenvalues of *A* have negative real parts. In GeneSPIDER2, network stability is characterized by the Interampatteness (IAA) degree, which relates to the spectral properties of the network matrix (5).

The stabilization procedure in GeneSPIDER2 adjusts the weights of the network to achieve a desired IAA level while preserving the network topology. This is accomplished through an iterative algorithm that scales the edge weights until the maximum real part of the eigenvalues falls within a specified range. Three stability levels are available: low (more stable, slower dynamics), medium, and high (less stable, faster dynamics).

#### **Scale-Free Network Generation**

Biological gene regulatory networks typically exhibit scale-free topology, characterized by a power-law degree distribution where a few hub genes have many connections while most genes have few connections (7). GeneSPIDER2 generates scale-free networks using the Barabási-Albert preferential attachment model (8). For large-scale networks, the toolbox employs a novel stitching algorithm that combines multiple small scale-free subnetworks (subGRNs).

The probability *P* of creating a connection to a node with degree *x* during the stitching process is given by:

[![][image2]](https://www.codecogs.com/eqnedit.php?latex=%20P%20%3D%20cx%5E%7B%5Calpha%7D%20#0)  
where *c* is a normalization constant and α is the exponent of the power-law distribution, which can be adjusted to match the properties of known biological networks (6). This approach enables the generation of large GRNs (thousands of genes) with controllable modularity while maintaining computational efficiency and network stability.

#### **Signal-to-Noise Ratio and Data Simulation**

GeneSPIDER2 simulates noisy gene expression data by adding Gaussian noise to the noise-free expression matrix. The Signal-to-Noise Ratio (SNR) is a critical parameter that controls the amount of noise in the simulated data. The standard deviation of the noise (σE) is calculated based on the desired SNR:

[![][image3]](https://www.codecogs.com/eqnedit.php?latex=%20%5Csigma_E%20%3D%20%5Cfrac%7Bs_g%7D%7BSNR%20%5Ccdot%20%5Csqrt%7B%5Cchi%5E2_%7B1-%5Calpha%7D\(ge\)%7D%7D%20#0)  
where *sg* is the *g*\-th singular value of the noise-free expression matrix *X*, α is the significance level (typically 0.05), and χ21-α(*ge*) is the chi-square distribution quantile with *ge* degrees of freedom (5). The noisy expression data is then obtained as:

[![][image4]](https://www.codecogs.com/eqnedit.php?latex=%20Y%20%3D%20X%20%2B%20E%20#0)  
where *E* ∈ ℝ*g*×*e* is a matrix of independent Gaussian noise with standard deviation σE.

#### **Single-Cell Data Simulation**

Single-cell RNA-seq data presents unique characteristics that differ from bulk RNA-seq, including lower read counts, higher technical noise, and the presence of dropout events where genes fail to be detected despite being expressed. GeneSPIDER2 models these features through a multi-step simulation process.

First, a simulated control count (SCC) matrix *M*SCC is generated to represent baseline expression levels. The mean count for each gene (μNB) is drawn from a negative binomial distribution with probability *p*NB and number of successes *R* \= 1:

[![][image5]](https://www.codecogs.com/eqnedit.php?latex=%20%5Cmu_%7BNB%7D%20%5Csim%20NB\(R%3D1%2C%20p_%7BNB%7D\)%20#0)  
The count distribution for each gene across cells is then drawn from a lognormal distribution using μNB and the standard deviation of the fold-change data (6). To create a clustered data structure reflecting cellular heterogeneity, cells are assigned to clusters, and cluster-specific means (μ1 for within-cluster and μ2 for outside-cluster, where μ1 \> μ2) are used to modulate expression levels.

The raw UMI counts are obtained by combining the fold-change data with the SCC matrix:

[![][image6]](https://www.codecogs.com/eqnedit.php?latex=%20Y_%7BUMI%7D%20%3D%20Y%20%5Codot%20M_%7BSCC%7D%20#0)  
where ⊙ denotes element-wise multiplication after taking the inverse logarithm of *Y*.

Finally, dropout events are simulated using a probabilistic model. The dropout probability for a given expression level μ is given by:

[![][image7]](https://www.codecogs.com/eqnedit.php?latex=%20P_%7Bdropout%7D\(%5Cmu\)%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20%5Cphi%20%5Cmu%7D%20#0)  
where φ is a dispersion parameter controlling the variance (6). A binary dropout matrix *E*D is constructed by drawing uniform random numbers and comparing them to the dropout probability. The final single-cell data with dropouts is:

[![][image8]](https://www.codecogs.com/eqnedit.php?latex=%20Y_%7BSC%7D%20%3D%20Y_%7BUMI%7D%20%5Codot%20E_D%20#0)  
\# Some final words

3. ### **Overview of the Protocol**

This protocol is organized into four main sections following this introduction. The Materials section describes the software requirements, hardware recommendations, and data structures used by GeneSPIDER2. The Methods section provides step-by-step procedures for the entire workflow of generating GRNs, simulating gene expression data, performing GRN inference, and benchmarking inference performance locally or on the online GRN Benchmark platform. The Notes section contains important technical considerations, troubleshooting tips, and best practices for using GeneSPIDER2 effectively.

2. ## **Materials**

   1. ### **Software and Hardware Requirements**

#### **Primary Software**

1\. MATLAB (The MathWorks, Natick, MA, USA): Version R2019b or later is required. GeneSPIDER2 has been tested on MATLAB R2019b through R2023a. The toolbox requires the following MATLAB toolboxes:

\- Statistics and Machine Learning Toolbox  
\- Optimization Toolbox  
\- Parallel Computing Toolbox (optional, for accelerated computations)

2\. GeneSPIDER2: Available from Bitbucket at https://bitbucket.org/sonnhammergrni/genespider or via DOI: 10.5281/zenodo.10949060. The software is distributed under the GPLv3 license (6).

#### **Optional Software**

1\. R (R Foundation for Statistical Computing, Vienna, Austria): Version 4.0.0 or later, for visualization using igraph package (only required for Section 3.8).

2\. Python: Version 3.7 or later with Seurat package through reticulate (for advanced single-cell data visualization).

#### **Hardware Requirements**

1\. Minimum requirements:

\- Processor: Multi-core CPU (Intel Core i5 or equivalent)  
\- RAM: 8 GB for networks up to 100 genes; 16 GB for networks up to 500 genes; 32 GB or more for networks with 1,000+ genes  
\- Storage: 10 GB free disk space for software and data

2\. Recommended specifications:

\- Processor: Multi-core CPU (Intel Core i7 or AMD Ryzen 7, 8+ cores)  
\- RAM: 64 GB for optimal performance with large-scale networks  
\- Storage: Solid-state drive (SSD) with 50+ GB free space

2. ### **Installation and Setup**

#### **Installing GeneSPIDER2**

1\. Download GeneSPIDER2 from the Bitbucket repository (https://bitbucket.org/sonnhammergrni/genespider) or Zenodo (DOI: 10.5281/zenodo.10949060).

2\. Extract the downloaded archive to a directory of your choice (e.g., C:\\MATLAB\\GeneSPIDER2 on Windows or \~/MATLAB/GeneSPIDER2 on macOS/Linux).

3\. Open MATLAB and navigate to the GeneSPIDER2 directory.

4\. Add GeneSPIDER2 and all subdirectories to the MATLAB path:

clear all;  
addpath(genpath('path/to/GeneSPIDER2'));  
    

Replace 'path/to/GeneSPIDER2' with the actual path to your GeneSPIDER2 installation.

5\. Verify the installation by checking that key functions are accessible:

which datastruct.scalefree2  
which Methods.LSCON  
    

These commands should return the full path to the respective functions. If they return an error, ensure the path was added correctly.

**Note:** If you plan to use GeneSPIDER2 regularly, add the addpath command to your MATLAB startup file (startup.m) so that GeneSPIDER2 is available in every MATLAB session. 

3. ### **Data Structures**

#### **Network Matrix (A)**

The network adjacency matrix is a square matrix where rows represent target genes and columns represent regulator genes. Non-zero elements indicate regulatory interactions. Example structure for a 5-gene network:

          G1    G2     G3     G4     G5  
     G1  0.0   0.0    0.0    0.0    0.9    
     G2 \-0.5   0.0    0.25   0.0   \-0.7    
     G3 \-0.9   0.0    0.0    0.0    0.0    
     G4  0.0   0.0   \-0.8    0.0    0.0    
     G5  0.0   0.0    0.56   0.77   0.0    
      
    

See Figure GRNex for a rendering of this example GRN.

#### 

#### **Perturbation Matrix (P)**

The perturbation design matrix specifies which genes are perturbed in each experiment or cell. For knockdown experiments, the matrix typically contains \-1 for perturbed genes and 0 for non-perturbed genes. For *g* genes with two replicates per gene, the matrix has dimensions *g* × 2*g*:

    % P \= \-\[eye(g) eye(g)\];  
    % This creates a matrix where each gene is knocked down in two experiments  
    

#### **Expression Matrix (Y)**

The gene expression matrix contains measured (or simulated) expression values. For bulk data, values typically represent log fold-changes relative to control. For single-cell data, values represent UMI counts (raw counts) or normalized expression values. Matrix dimensions are *g* × *e*, where *e* is the number of experiments or cells.

#### **Dataset Object (D)**

GeneSPIDER2 uses a structured data object to organize all information needed for inference and analysis. The Dataset object contains the following fields:

    D.network   % The true network adjacency matrix (for synthetic data)  
    D.E         % Noise matrix (output noise)  
    D.F         % Input noise matrix  
    D.Y         % Gene expression matrix  
    D.P         % Perturbation design matrix  
    D.lambda    % Noise variance parameters \[σ²\_Y, σ²\_P\]  
    D.cvY       % Covariance matrix of output noise  
    D.cvP       % Covariance matrix of input noise  
    D.sdY       % Standard deviation of output noise  
    D.sdP       % Standard deviation of input noise  
    

4. ### **Example Datasets**

GeneSPIDER2 includes example datasets for testing and learning: \#Check and give more details

1\. Synthetic bulk data: 50-gene scale-free network with knockdown perturbations

2\. Real bulk data: K562 cell line expression data (Y\_bulk\_k562.mat, P\_bulk\_k562.mat)

3\. Real single-cell data: K562 cell line Perturb-seq data (Y\_sc\_k562.mat, P\_sc\_k562.mat)

4\. GRN Benchmark data: Networks from GeneNetWeaver and GeneSPIDER with varying noise levels

3. ## **Methods**

This section provides step-by-step procedures with code snippets for: (i) generating realistic GRNs, (ii) simulating bulk gene expression data with perturbations, (iii) simulating single-cell RNA-seq data with perturbations, (iv) performing GRN inference using various methods, (v) benchmarking inference performance, (vi) working with experimental data, and (vii) preparing data for the online GRN Benchmark platform. 

1. ### **Generating Synthetic Scale-Free Gene Regulatory Networks**

#### **Creating a Small Scale-Free Network**

The following procedure generates a small-scale GRN (50-100 genes) with scale-free topology and ensures network stability.

**Procedure:**

1\. Clear the workspace and add GeneSPIDER2 to the path:

clear all;  
addpath(genpath('path/to/GeneSPIDER2'));  
    

2\. Define the network size (number of genes):

N \= 50;  % Network size (number of genes)  
    

3\. Specify the desired average sparsity (average number of edges per node):

S \= 3;  % Average of 3 edges per node  
    

4\. Generate the scale-free network topology:

A \= datastruct.scalefree2(N, S);  
      
This function uses the Barabási-Albert preferential attachment model to create a scale-free network. The resulting matrix A contains random weights drawn from a uniform distribution.

5\. Stabilize the network to ensure biological feasibility:

A \= datastruct.stabilize(A, 'iaa', 'low');  
    

The stabilization procedure adjusts edge weights while preserving topology. The 'low' IAA setting produces more stable dynamics. Other options are 'medium' and 'high'.

6\. Create a Network object:

Net \= datastruct.Network(A, 'myNetwork');  
      
The Network object encapsulates the adjacency matrix and computes derived properties including the static gain matrix (Net.G).

7\. Examine network properties:

% View the adjacency matrix  
disp('Network adjacency matrix:');  
disp(A);

% Check network stability (all eigenvalues should have negative real parts)  
eigvals \= eig(A);  
disp('Eigenvalues of network matrix:');  
disp(eigvals);  
disp(\['Maximum real part: ', num2str(max(real(eigvals)))\]);

% Calculate degree distribution  
in\_degree \= sum(A \~= 0, 2);   % In-degree (column sum)  
out\_degree \= sum(A \~= 0, 1)';  % Out-degree (row sum)

disp(\['Average in-degree: ', num2str(mean(in\_degree))\]);  
disp(\['Average out-degree: ', num2str(mean(out\_degree))\]);  
      
**Note:** The stabilization process may not converge for all random network topologies, particularly those with very high connectivity. If stabilization fails, try generating a new network or adjusting the sparsity parameter. 

#### 

#### **Creating Large-Scale Networks**

For networks with hundreds to thousands of genes, use the large-scale network generation function.

**Procedure:**

1\. Define the network size and sparsity:

m \= 1000;      % Network size (500 genes)  
Savg \= 3;     % Average sparsity  
    

2\. Generate the large-scale network:

A \= datastruct.large\_scalefree(m, Savg);  
    

This function employs the subGRN stitching algorithm to efficiently generate large networks. The resulting network exhibits scale-free topology and modular structure (6).

3\. Create a Network object and verify properties:

Net \= datastruct.Network(A, 'large\_network');

% Check basic properties  
fprintf('Network size: %d genes\\n', size(A, 1));  
fprintf('Number of edges: %d\\n', nnz(A));  
fprintf('Average degree: %.2f\\n', nnz(A) / size(A, 1));  
    

### See **Figure large\_scalefree** for a rendering of the generated GRN.

**Note 3:** Large-scale network generation can take 10-20 minutes for 20,000 genes depending on computer specifications. The large\_scalefree function automatically ensures network stability, so additional stabilization is not required. 

2. ### **Simulating Bulk Gene Expression Data with Perturbations**

#### **Generating Synthetic Bulk Data**

This procedure simulates gene expression data from knockdown perturbation experiments.

**Procedure:**

1\. Create or load a network (following Section 3.1):

N \= 50;  
S \= 3;  
A \= datastruct.scalefree2(N, S);  
A \= datastruct.stabilize(A, 'iaa', 'low');  
Net \= datastruct.Network(A, 'myNetwork');  
      
2\. Define the perturbation design matrix. For two replicates of each gene knockdown:

P \= \-\[eye(N) eye(N)\];  % Knockdown each gene twice  
    

3\. Calculate the noise-free expression matrix using the static gain:

X \= Net.G \* P;  % G is the static gain matrix (inverse of A)  
    

4\. Define the signal-to-noise ratio:

SNR \= 0.1;  % 0.1 \= medium noise, 1 \= low noise, 0.01 \= high noise  
    

5\. Calculate the noise standard deviation:

s \= svd(X);  % Singular value decomposition  
stdE \= s(N) / (SNR \* sqrt(chi2inv(1 \- analyse.Data.alpha, numel(P))));  
      
The variable analyse.Data.alpha is typically set to 0.05 (95% confidence level).

6\. Generate noise matrices:

E \= stdE \* randn(size(P));  % Output noise (measurement noise)  
F \= zeros(size(P));         % Input noise (assumed zero for knockdowns)  
      
7\. Construct the Dataset object:

D(1).network \= Net.network;     % Store the true network  
D(1).E \= E;                     % Noise matrix  
D(1).F \= F;                     % Input noise matrix  
D(1).Y \= X \+ E;                 % Noisy expression data  
D(1).P \= P;                     % Perturbation design  
D(1).lambda \= \[stdE^2, 0\];      % Noise variances  
D(1).cvY \= D.lambda(1) \* eye(N);  % Covariance of output noise  
D(1).cvP \= zeros(N);            % Covariance of input noise  
D(1).sdY \= stdE \* ones(size(D.P));  % Standard deviation of output noise  
D(1).sdP \= zeros(size(D.P));    % Standard deviation of input noise  
      
8\. Create the Dataset object:

Data \= datastruct.Dataset(D, Net);  
      
9\. Save the data (optional):

save('synthetic\_bulk\_data.mat', 'Data', 'Net', 'A', 'P');  
    

3. ### **Simulating Single-Cell RNA-seq Data with Perturbations**

#### **Generating Synthetic Single-Cell Data**

This procedure simulates perturbed single-cell RNA-seq data with realistic noise characteristics including dropouts and count distributions.

**Procedure:**

1\. Create or load a large-scale network:

m \= 500;      % Number of genes  
Savg \= 3;     % Average sparsity  
A \= datastruct.large\_scalefree(m, Savg);  
    

2\. Define the perturbation design for single-cell experiments. For 20 cells per gene perturbation:

cn \= 20;  % Number of cells per gene perturbation  
P \= \-repmat(eye(m), 1, cn);  % Total of m \* cn cells  
    

3\. Set the noise level:

SNRv \= 0.1;  % Signal-to-noise ratio  
    

4\. Generate single-cell data using the scdata function:

\[Y, X, Ed, Eg, SCC\] \= datastruct.scdata(A, P, 'SNR', SNRv, 'raw\_counts', false);  
    

This function performs the complete single-cell data simulation pipeline including:

\- Y: Final single-cell expression data (with dropouts)  
\- X: Noise-free fold-change expression  
\- Ed: Dropout noise matrix  
\- Eg: Gaussian noise matrix  
\- SCC: Simulated control count matrix

For raw UMI count data instead of fold-change data, set 'raw\_counts' to true:

\[Y, X, Ed, Eg, SCC\] \= datastruct.scdata(A, P, 'SNR', SNRv, 'raw\_counts', true);  
    

5\. Construct the Dataset object for single-cell data:

D(1).network \= A;  
D(1).E \= Ed .\* Eg;      % Total noise (dropouts and Gaussian)  
D(1).F \= zeros(size(P));  
D(1).Y \= Y;             % Single-cell expression data  
D(1).P \= P;             % Perturbation design matrix

% Calculate noise parameters  
stdE \= sqrt(var(X(:)) / SNRv);  
D(1).lambda \= \[stdE^2, 0\];  
D(1).cvY \= D.lambda(1) \* eye(m);  
D(1).cvP \= zeros(m);  
D(1).sdY \= stdE \* ones(size(D.P));  
D(1).sdP \= zeros(size(D.P));  
    

See **Figure scdata** with a visualization of the simulated single-cell data.

6\. Create the Network and Dataset objects:

Net \= datastruct.Network(A, 'scNetwork');  
Data \= datastruct.Dataset(D, Net);  
    

7\. Examine single-cell data properties:

% Calculate mean expression and variance  
mean\_expr \= mean(Y, 2);  
var\_expr \= var(Y, 0, 2);

% Calculate dropout rate per gene  
dropout\_rate \= sum(Y \== 0, 2\) / size(Y, 2);

fprintf('Mean expression range: \[%.2f, %.2f\]\\n', min(mean\_expr), max(mean\_expr));  
fprintf('Mean dropout rate: %.2f%%\\n', mean(dropout\_rate) \* 100);  
    

### 

**Note 4:** The scdata function includes many optional parameters for fine-tuning single-cell data properties including cluster number, dispersion parameter (φ), negative binomial probability (*p*NB), and cluster separation strength. See the GeneSPIDER2 documentation for complete parameter descriptions. 

### 

4. ### **Gene Regulatory Network Inference**

#### **Basic Network Inference**

GeneSPIDER2 includes multiple inference methods that can exploit perturbation information. This section demonstrates the LSCON (Least Squares Cut-Off with Normalization) method, which has shown robust performance across various data types (5).

**Procedure:**

1\. Load or create a Dataset object (from Section 3.2 or 3.3).

2\. Define the regularization parameter range (ζ values):

zeta \= logspace(-3, 0, 30);  % 30 networks spanning the sparsity spectrum  
    

The regularization parameter ζ controls network sparsity. Lower values produce denser networks, while higher values produce sparser networks. Using a range of values allows exploration of the full precision-recall curve.

3\. Select an inference method:  \#Add list of all methods

infMethod \= 'LSCON';  % Other options: 'lasso', 'ridgeco', 'GENIE3', 'CLR', etc.  
    

4\. Perform network inference:

\[Aest, z\] \= Methods.(infMethod)(Data, zeta);  
    

The function returns:

\- Aest: A 3D array of inferred networks (g × g × length(zeta))  
\- z: The actual ζ values used (may differ from input)

5\. Select a specific network from the ζ spectrum:

Aest\_selected \= Aest(:, :, 25);  % Select the 25th network  
    

6\. Visualize the sparsity of the inferred networks (See **Figure sparsity**):

% Calculate sparsity for each ζ  
s \= size(Aest, 1);  
sparsity \= squeeze(sum(sum(Aest \~= 0, 1), 2)) / s

figure;  
semilogx(z, sparsity, 'LineWidth', 2);  
xlabel('Regularization parameter \\zeta');  
ylabel('Sparsity \- number of inferred edges per node');  
title('Sparsity of inferred GRN as a function of \\zeta');  
fontsize(14,"points")  
grid on;

#### 

#### **Inference from Experimental Data**

This procedure demonstrates how to perform GRN inference on real bulk RNA-seq data.

**Procedure:**

1\. Load experimental expression and perturbation data:

load('data/Y\_bulk\_k562.mat');  % Load expression data  
load('data/P\_bulk\_k562.mat');  % Load perturbation design  
Y \= Y\_bulk\_k562;  
P \= P\_bulk\_k562;  
      
2\. Determine the number of genes:

N \= size(Y, 1);  
    

3\. Create an empty network (since the true network is unknown):

A \= zeros(N);  
Net \= datastruct.Network(A, 'experimentalNetwork');  
    

4\. Estimate noise parameters from the data:

stdE \= std(Y(:));  % Estimate noise from expression variance  
    

5\. Construct the Dataset object:

D(1).network \= Net.network;  
D(1).E \= zeros(size(Y));     % Unknown true noise  
D(1).F \= zeros(size(P));  
D(1).Y \= Y;                  % Experimental expression data  
D(1).P \= P;                  % Experimental perturbation design  
D(1).lambda \= \[stdE^2, 0\];  
D(1).cvY \= D.lambda(1) \* eye(N);  
D(1).cvP \= zeros(N);  
D(1).sdY \= stdE \* ones(size(D.P));  
D(1).sdP \= zeros(size(D.P));

Data \= datastruct.Dataset(D, Net);  
    

6\. Perform inference:

zeta \= logspace(-6, 0, 30);  
infMethod \= 'LSCON';  
\[Aest, z\] \= Methods.(infMethod)(Data, zeta);  
    

7\. Save the inferred networks:

save('inferred\_networks\_k562.mat', 'Aest', 'z', 'Data');  
    

5. ### **Benchmarking GRN Inference Performance**

#### **Basic Performance Metrics**

When the true network is known (synthetic data), GeneSPIDER2 can compute standard performance metrics including AUROC, AUPR, precision, recall, and F1-score.

**Procedure:**

1\. Ensure you have both the true network and inferred networks:

% True network from Net object  
A\_true \= Net.A;

% Inferred networks from inference procedure  
% Aest is g × g × n\_zeta array  
    

2\. Create a CompareModels object:

M \= analyse.CompareModels(Net, Aest);  
    

This function computes performance metrics for all networks in the ζ spectrum.

3\. Display key performance metrics:

% Area Under Receiver Operating Characteristic curve  
fprintf('AUROC: %.4f\\n', M.AUROC);

% Area Under Precision-Recall curve  
fprintf('AUPR: %.4f\\n', M.AUPR);

% Maximum F1-score across ζ values  
fprintf('Max F1-score: %.4f\\n', max(M.F1));

% Find the ζ index with maximum F1-score  
\[max\_f1, best\_idx\] \= max(M.F1);  
fprintf('Best ζ index: %d (ζ \= %.6f)\\n', best\_idx, z(best\_idx));  
    

4\. Visualize performance curves (See **Figure AUcurves**):

figure('Position', \[100, 100, 1200, 400\]);

% Precision-Recall curve  
subplot(1, 3, 1);  
plot(M.sen, M.pre, 'LineWidth', 2);  
xlabel('Recall');  
ylabel('Precision');  
title(\['Precision-Recall Curve (AUPR \= ', num2str(M.AUPR, '%.3f'), ')'\]);  
grid on;  
axis(\[0 1 0 1\]);

% ROC curve  
subplot(1, 3, 2);  
plot(M.comspe, M.sen, 'LineWidth', 2);  
hold on;  
plot(\[0 1\], \[0 1\], 'k--');  % Diagonal reference line  
xlabel('False Positive Rate');  
ylabel('True Positive Rate');  
title(\['ROC Curve (AUROC \= ', num2str(M.AUROC, '%.3f'), ')'\]);  
grid on;  
axis(\[0 1 0 1\]);

% F1-score vs. regularization parameter  
subplot(1, 3, 3);  
semilogx(z, M.F1, 'LineWidth', 2);  
xlabel('Regularization parameter \\zeta');  
ylabel('F1-score');  
title(\['F1-score vs. \\zeta (Max F1 \= ', num2str(max(M.F1), '%.3f'), ')'\]);  
grid on;  
    

5\. Extract the best network:

Aest\_best \= Aest(:, :, best\_idx);

% Compare edge statistics  
true\_edges \= nnz(A\_true);  
inferred\_edges \= nnz(Aest\_best);  
true\_pos \= nnz((A\_true \~= 0\) & (Aest\_best \~= 0));

fprintf('True network edges: %d\\n', true\_edges);  
fprintf('Inferred network edges: %d\\n', inferred\_edges);  
fprintf('True positives: %d\\n', true\_pos);  
fprintf('Precision: %.4f\\n', true\_pos / inferred\_edges);  
fprintf('Recall: %.4f\\n', true\_pos / true\_edges);  
    

#### 

#### **Comparing Multiple Inference Methods**

This procedure compares the performance of different inference algorithms on the same dataset.

**Procedure:**

1\. Define a list of inference methods to compare:

methods \= {'LSCON', 'lasso', 'ridgeco', 'GENIE3', 'CLR'};  
n\_methods \= length(methods);  
    

2\. Initialize storage for results:

results \= struct();  
for i \= 1:n\_methods  
    results(i).method \= methods{i};  
end  
    

3\. Perform inference with each method:

zeta \= logspace(-3, 0, 30);

for i \= 1:n\_methods  
    fprintf('Running %s...\\n', methods{i});  
    tic;  
    \[Aest, z\] \= Methods.(methods{i})(Data, zeta);  
    results(i).time \= toc;  
    results(i).Aest \= Aest;  
    results(i).z \= z;  
      
    % Compute performance metrics  
    M \= analyse.CompareModels(Net, Aest);  
    results(i).AUROC \= M.AUROC;  
    results(i).AUPR \= M.AUPR;  
    results(i).maxF1 \= max(M.F1);  
    results(i).Precision \= M.pre;  
    results(i).Recall \= M.sen;  
    results(i).F1 \= M.F1;  
      
    fprintf('  AUROC: %.4f, Max F1: %.4f, Time: %.2f s\\n', ...  
            M.AUROC, max(M.F1), results(i).time);  
end  
    

4\. Create a comparative visualization (See **Figure multiMethods**):

figure('Position', \[100, 100, 1000, 800\]);

% AUROC comparison  
subplot(2, 2, 1);  
auroc\_values \= \[results.AUROC\];  
bar(auroc\_values);  
set(gca, 'XTickLabel', methods);  
ylabel('AUROC');  
title('AUROC Comparison');  
ylim(\[0 1\]);  
grid on;

% Max F1-score comparison  
subplot(2, 2, 2);  
f1\_values \= \[results.maxF1\];  
bar(f1\_values);  
set(gca, 'XTickLabel', methods);  
ylabel('Max F1-score');  
title('F1-score Comparison');  
ylim(\[0 1\]);  
grid on;

% Precision-Recall curves  
subplot(2, 2, 3);  
hold on;  
colors \= lines(n\_methods);  
for i \= 1:n\_methods  
    plot(results(i).Recall, results(i).Precision, ...  
         'LineWidth', 2, 'Color', colors(i, :), ...  
         'DisplayName', methods{i});  
end  
xlabel('Recall');  
ylabel('Precision');  
title('Precision-Recall Curves');  
legend('Location', 'best');  
grid on;  
axis(\[0 1 0 1\]);

% Computation time comparison  
subplot(2, 2, 4);  
time\_values \= \[results.time\];  
bar(time\_values);  
set(gca, 'XTickLabel', methods);  
ylabel('Time (seconds)');  
title('Computation Time');  
grid on;  
    

6. ### **Nested Bootstrap Analysis for FDR Control  \#remove?**

#### **Running Nested Bootstrap**

The nested bootstrap procedure controls the false discovery rate (FDR) by comparing network inference results on real data with results on shuffled (null) data (9). This approach provides statistical confidence in the inferred edges.

**Procedure:**

1\. Prepare the Dataset object (synthetic or experimental).

2\. Define nested bootstrap parameters:

method \= "lscon";         % Inference method  
nest \= 10;                % Number of outer iterations (bootstrap samples)  
boot \= 10;                % Number of inner iterations (network realizations)  
zetavec \= logspace(-6, 0, 30);  % Regularization parameters  
fdr \= 0.05;               % False discovery rate threshold  
paral \= true;             % Enable parallel computation  
cornr \= 4;                % Number of CPU cores to use  
direc \= '\~/genespider\_results/';  % Output directory  
      
3\. Execute the nested bootstrap:

nbout \= Methods.NestBoot(Data, method, nest, boot, zetavec, fdr, direc, paral, cornr);  
      
This function performs the following steps:

\- Creates nest bootstrap samples of the data  
\- For each bootstrap sample, performs boot network inferences  
\- Computes support statistics for each edge  
\- Compares real data support to shuffled data support  
\- Determines FDR-controlled edge sets

**Note 5:** Nested bootstrap is computationally intensive. For a 50-gene network with nest=10 and boot=10, expect approximately 30-60 minutes runtime on a modern multi-core system. Use parallel computation to reduce runtime.   
4\. Extract results from the output structure:

% Binary networks at FDR cutoff  
binary\_networks \= nbout.binary\_networks;

% Signed networks (with activation/repression information)  
signed\_networks \= nbout.signed\_networks;

% Support threshold corresponding to desired FDR  
support\_threshold \= nbout.bin\_cutoff;

fprintf('Support threshold for FDR=%.2f: %.4f\\n', fdr, support\_threshold);  
    

5\. Visualize support distributions:

figure('Position', \[100, 100, 1200, 400\]);

% Accumulated frequency plot  
subplot(1, 3, 1);  
plot(nbout.accumulated\_frequency, 'LineWidth', 2);  
xlabel('Support bins');  
ylabel('Accumulated frequency');  
title('Support Distribution');  
grid on;

% Binned frequency (histogram)  
subplot(1, 3, 2);  
bar(nbout.binned\_frequency);  
xlabel('Support bins');  
ylabel('Frequency');  
title('Support Histogram');  
grid on;

% FPR crossing point  
subplot(1, 3, 3);  
plot(nbout.FP\_rate\_cross, 'LineWidth', 2);  
hold on;  
plot(\[support\_threshold support\_threshold\], ylim, 'r--', 'LineWidth', 2);  
xlabel('Support threshold');  
ylabel('False positive rate');  
title(\['FDR Control (threshold \= ', num2str(support\_threshold, '%.3f'), ')'\]);  
legend('FPR curve', 'FDR cutoff');  
grid on;  
      
6\. If the true network is known, evaluate nested bootstrap performance:

% Select the FDR-controlled network  
A\_fdr \= signed\_networks(:, :, end);  % Network at FDR cutoff

% Compare to true network  
M\_fdr \= analyse.CompareModels(Net.A, A\_fdr);

fprintf('Nested Bootstrap Performance (FDR=%.2f):\\n', fdr);  
fprintf('  Precision: %.4f\\n', M\_fdr.Precision);  
fprintf('  Recall: %.4f\\n', M\_fdr.Recall);  
fprintf('  F1-score: %.4f\\n', M\_fdr.F1);  
    

7. ### **Preparing Data for GRN Benchmark Platform**

#### **Converting GeneSPIDER2 Output to GRN Benchmark Format**

The GRN Benchmark (https://grnbenchmark.org/) is a web-based platform for standardized evaluation of GRN inference methods. This procedure converts GeneSPIDER2 inferred networks to the required format.

**Procedure:**

1\. Load data files from GRN Benchmark:

clear;  
addpath(genpath('path/to/GeneSPIDER2'));

tool \= "GeneNetWeaver";  % Options: "GeneSPIDER" or "GeneNetWeaver"  
nlev \= "LowNoise";       % Options: "LowNoise", "MediumNoise", "HighNoise"  
pathin \= 'path/to/GRNBenchmark/Data/';  
pathout \= 'path/to/output/folder/';  
reps \= 5;                % Number of replicate networks (usually 5\)  
      
2\. Process each network replicate:

for j \= 1:reps  
    fprintf('Processing network %d of %d...\\n', j, reps);  
      
    % Load expression data  
    Y\_file \= sprintf('%s%s\_%s\_Network%d\_GeneExpression.csv', ...  
                     pathin, tool, nlev, j);  
    Y \= readtable(Y\_file, 'ReadRowNames', true);  
    gnsnms \= string(Y.Properties.RowNames);  % Gene names  
    Y \= table2array(Y);  
      
    % Load perturbation design  
    P\_file \= sprintf('%s%s\_%s\_Network%d\_Perturbations.csv', ...  
                     pathin, tool, nlev, j);  
    P \= readtable(P\_file, 'ReadRowNames', true);  
    P \= table2array(P);  
      
    N \= size(Y, 1);  % Number of genes  
      
    % Create empty network (true network unknown for benchmarking)  
    A \= zeros(N);  
    Net \= datastruct.Network(A, 'benchmarkNetwork');  
      
    % Construct Dataset object  
    D(1).network \= \[\];  
    D(1).E \= zeros(size(Y));  
    D(1).F \= zeros(N);  
    D(1).Y \= Y;  
    D(1).P \= P;  
    D(1).lambda \= \[std(Y(:))^2, 0\];  
    D(1).cvY \= D.lambda(1) \* eye(N);  
    D(1).cvP \= zeros(N);  
    D(1).sdY \= std(Y(:)) \* ones(size(D.P));  
    D(1).sdP \= zeros(size(D.P));  
      
    Data \= datastruct.Dataset(D, Net);  
      
    % Perform inference (no cutoff, full network)  
    zeta \= 0;  % Return full network; GRN Benchmark applies cutoff internally  
    infMethod \= 'LSCON';  
    \[Aest, z\] \= Methods.(infMethod)(Data, zeta);  
      
    inet \= Aest;  % Inferred network  
      
    % Prepare edge list in GRN Benchmark format  
    % Convert to signed edges  
    wedges \= compose("%9.5f", round(inet(:), 5));  % Keep weights  
    inet(inet \< 0\) \= \-1;  % Negative regulation  
    inet(inet \> 0\) \= 1;   % Positive regulation  
    edges \= inet(:);      % Flatten to vector  
      
    % Create edge names (from/to gene pairs)  
    s \= size(inet, 1);  
    nams\_edges \= \[repmat(1:s, 1, s); repelem(1:s, s)\]';  
    edges\_from \= gnsnms(nams\_edges(:, 2));  % Regulator (column)  
    edges\_to \= gnsnms(nams\_edges(:, 1));    % Target (row)  
    nrid \= string((1:length(edges\_from))');  
      
    % Create table with required format  
    edge\_list \= table(nrid, edges\_from, edges\_to, wedges, string(edges));  
    allVars \= 1:width(edge\_list);  
    newNames \= \["ID", "Regulator", "Target", "Weight", "Sign"\];  
    edge\_list \= renamevars(edge\_list, allVars, newNames);  
      
    % Add header row  
    Var1 \= "";  
    Var2 \= "Regulator";  
    Var3 \= "Target";  
    Var4 \= "Weight";  
    Var5 \= "Sign";  
    newNamesTab \= table(Var1, Var2, Var3, Var4, Var5);  
    newNamesTab \= renamevars(newNamesTab, allVars, newNames);  
      
    % Remove zero edges and combine with header  
    edge\_list(edges \== 0, :) \= \[\];  
    edge\_list2 \= \[newNamesTab; edge\_list\];  
      
    % Save to CSV file  
    out\_file \= sprintf('%s%s\_%s\_Network%d\_grn.csv', pathout, tool, nlev, j);  
    writetable(edge\_list2, out\_file, 'QuoteStrings', true, 'WriteVariableNames', false);  
      
    fprintf('  Saved %s\\n', out\_file);  
end

fprintf('All networks processed successfully.\\n');  
    

**Note 6:** The GRN Benchmark format requires specific column names and structure. The Regulator column indicates the source gene (from column in GeneSPIDER2), and the Target column indicates the destination gene (from row in GeneSPIDER2). The Sign column contains \-1 for repression, 1 for activation, and the Weight column contains the continuous edge weight. 

8. ### **Visualization of Inferred Networks**

#### Network Visualization in Matlab (See Figure InfGRN):

% Select a network to visualize  
Aest\_viz \= Aest(:, :, 25);  % Choose a sparse network in zeta spectrum

G \= digraph(Aest\_viz’, 'OmitSelfLoops');

figure('Position', \[100 100 800 600\]);  
h \= plot(G, 'Layout', 'force', ...  
         'ArrowSize', 12, ...  
         'NodeColor', 'c', ...  
         'MarkerSize', 10, ...  
         'LineWidth', 1.5);

% Color edges by regulation type  
weights \= G.Edges.Weight;  
edge\_colors \= zeros(numedges(G), 3);  
edge\_colors(weights \< 0, 1\) \= 1;  % Red for repression  
edge\_colors(weights \> 0, 3\) \= 1;  % Blue for activation  
h.EdgeColor \= edge\_colors;

title('Gene Regulatory Network Inferred by ', infMethod);  
xlabel('Red \= Repression, Blue \= Activation');    

#### 

#### **Advanced Visualization: Heatmap of Network Adjacency Matrix**

For dense networks or comparative analysis, heatmap visualization is often more informative than node-edge diagrams.

**Procedure in MATLAB:**

% Select a network to visualize  
Aest\_viz \= Aest(:, :, 23);  % Choose 23rd network in zeta spectrum

% Create heatmap  
figure('Position', \[100, 100, 800, 700\]);  
imagesc(Aest\_viz);  
colormap(redblue);  % Red for positive, blue for negative  
colorbar;  
caxis(\[-max(abs(Aest\_viz(:))), max(abs(Aest\_viz(:)))\]);  % Symmetric color scale

% Add labels  
xlabel('Regulator genes (columns)');  
ylabel('Target genes (rows)');  
title('Inferred GRN Adjacency Matrix');

% Add grid for better readability  
hold on;  
for i \= 1:size(Aest\_viz, 1\)  
    plot(\[0.5, size(Aest\_viz, 2)+0.5\], \[i+0.5, i+0.5\], 'k-', 'LineWidth', 0.1);  
    plot(\[i+0.5, i+0.5\], \[0.5, size(Aest\_viz, 1)+0.5\], 'k-', 'LineWidth', 0.1);  
end  
hold off;  
axis equal tight;  
    

#### 

#### **Comparative Network Visualization**

When comparing true and inferred networks, it is useful to visualize true positives, false positives, and false negatives.

**Procedure:**

% Binarize networks for comparison  
A\_true\_bin \= (Net.A \~= 0);  
Aest\_bin \= (Aest\_best \~= 0);

% Classify edges  
TP \= A\_true\_bin & Aest\_bin;      % True positives  
FP \= \~A\_true\_bin & Aest\_bin;     % False positives  
FN \= A\_true\_bin & \~Aest\_bin;     % False negatives  
TN \= \~A\_true\_bin & \~Aest\_bin;    % True negatives

% Create comparison matrix  
% 0=TN, 1=FP, 2=FN, 3=TP  
comparison \= zeros(size(A\_true\_bin));  
comparison(FP) \= 1;  
comparison(FN) \= 2;  
comparison(TP) \= 3;

% Visualize  
figure('Position', \[100, 100, 1200, 400\]);

subplot(1, 3, 1);  
imagesc(A\_true\_bin);  
colormap(gca, gray);  
title('True Network');  
xlabel('Regulator');  
ylabel('Target');  
axis equal tight;

subplot(1, 3, 2);  
imagesc(Aest\_bin);  
colormap(gca, gray);  
title('Inferred Network');  
xlabel('Regulator');  
ylabel('Target');  
axis equal tight;

subplot(1, 3, 3);  
imagesc(comparison);  
cmap \= \[1 1 1; 1 0.5 0.5; 0.5 0.5 1; 0.5 1 0.5\];  % White, Red, Blue, Green  
colormap(gca, cmap);  
title('Comparison (Green=TP, Red=FP, Blue=FN)');  
xlabel('Regulator');  
ylabel('Target');  
axis equal tight;  
colorbar('Ticks', \[0.375, 1.125, 1.875, 2.625\], ...  
         'TickLabels', {'TN', 'FP', 'FN', 'TP'});  
    

## **4\. Notes**  \#should to be integrated in text

### **4.1 General Considerations**

**Note 7: Understanding edge directionality.** In GeneSPIDER2, regulatory relationships are represented with the direction from column to row in the adjacency matrix. This means that *Aij* represents the effect of gene *j* (column) on gene *i* (row). This convention is opposite to some other network representations and must be carefully considered when interpreting results or converting to other formats.

**Note 8: Choice of inference method.** GeneSPIDER2 includes over 20 inference methods divided into two categories: methods that exploit perturbation information (P-matrix methods) and methods that do not. For perturbation-based experiments, P-matrix methods (LSCON, lasso, ridgeco, etc.) generally outperform non-perturbation methods (CLR, ARACNE, GENIE3) (5, 10). However, the optimal method depends on data characteristics including sample size, noise level, and network properties. We recommend benchmarking multiple methods on pilot data.

**Note 9: Regularization parameter selection.** The regularization parameter ζ controls the trade-off between network sparsity and fitting error. In practice, the optimal ζ value depends on the true (unknown) network structure. Using a range of ζ values and computing the full precision-recall curve provides more information than selecting a single network. For experimental data where the true network is unknown, consider using nested bootstrap analysis to select edges with controlled FDR rather than choosing a single ζ value.

**Note 10: Computational requirements.** Memory requirements scale approximately with *O*(*g*2) for network storage and *O*(*ge*) for expression data storage, where *g* is the number of genes and *e* is the number of experiments/cells. For large-scale networks (\>1000 genes), ensure sufficient RAM is available. Computation time for network generation scales approximately linearly with network size using the large-scale algorithm, while inference time depends on the method, with LSCON being among the fastest perturbation-based methods.

### **4.2 Network Generation and Stability**

**Note 11: Scale-free network properties.** Real biological GRNs exhibit power-law degree distributions with exponents typically in the range 1.5-2.5 (7). The GeneSPIDER2 default parameters produce networks with properties similar to known biological networks. However, users can adjust the power-law exponent α and other parameters to match specific biological systems. The modularity of generated networks can be controlled through the subGRN size parameter in large-scale generation (6).

**Note 12: Stability issues.** If the stabilization procedure fails to converge, this typically indicates that the random network topology is fundamentally unstable. Solutions include: (i) reducing the average sparsity parameter, (ii) regenerating the network with a different random seed, or (iii) using the large-scale generation function which includes automatic stabilization. Note that some network topologies (particularly those with strong positive feedback loops) cannot be stabilized while maintaining the original structure.

**Note 13: Sign distribution.** By default, GeneSPIDER2 generates approximately 62% activating edges and 38% repressing edges, based on statistics from the TRRUST database (11). This ratio can be adjusted using optional parameters in the network generation functions. The balance between activation and repression affects network stability and dynamics.

### **4.3 Data Simulation**

**Note 14: Choice of SNR.** The signal-to-noise ratio profoundly impacts inference difficulty. Typical values are: SNR \= 1 (low noise, easy inference), SNR \= 0.1 (medium noise, moderate difficulty), SNR \= 0.01 (high noise, difficult inference). These values are based on the ratio of signal variance to noise variance. Real experimental data typically falls in the range SNR \= 0.05-0.5 depending on the technology and biological system (5).

**Note 15: Single-cell data parameters.** The scdata function includes many parameters for controlling single-cell data properties. Key parameters include: (i) raw\_counts (true/false) determines whether to return UMI counts or log fold-changes, (ii) the dispersion parameter φ controls dropout probability (higher φ \= fewer dropouts), (iii) p\_NB controls the mean-variance relationship in the negative binomial distribution, and (iv) cluster-related parameters control cellular heterogeneity. Default parameters produce data similar to real Perturb-seq experiments (6).

**Note 16: Perturbation design considerations.** The perturbation matrix P defines the experimental design. Common designs include: (i) single-gene knockdowns with P \= \-eye(g) (one experiment per gene), (ii) replicated knockdowns with P \= \-\[eye(g) eye(g)\] (two replicates per gene), (iii) partial knockdowns with intermediate values between 0 and \-1, and (iv) combinatorial perturbations with multiple genes perturbed simultaneously. The choice of design affects inference power, with more experiments generally improving accuracy but increasing cost.

### **4.4 Network Inference**

**Note 17: Handling missing values.** If experimental data contains missing values (NaN), these must be handled before inference. Options include: (i) removing genes or experiments with missing values, (ii) imputing missing values using mean, median, or more sophisticated methods, or (iii) using inference methods that can handle missing data. GeneSPIDER2 inference methods assume complete data and will produce errors or incorrect results with missing values.

**Note 18: Normalization of experimental data.** For real RNA-seq data, appropriate normalization is critical. Bulk RNA-seq data should be normalized (e.g., using DESeq2, edgeR, or similar methods) and converted to log fold-changes relative to control before loading into GeneSPIDER2. Single-cell data can be provided as either raw UMI counts or normalized log expression values. The choice affects noise estimation and should be documented.

**Note 19: Inference method parameters.** Each inference method in GeneSPIDER2 has specific parameters and assumptions. LSCON assumes linear relationships and Gaussian noise; lasso performs L1 regularization favoring sparse solutions; ridgeco uses L2 regularization for smoother solutions; GENIE3 uses tree-based methods that can capture non-linearities; CLR uses mutual information. Consult the GeneSPIDER2 documentation and original method publications for detailed parameter descriptions.

**Note 20: Parallel computation.** Several GeneSPIDER2 functions support parallel computation using MATLAB's Parallel Computing Toolbox. To enable parallel processing, create a parallel pool before running computationally intensive functions:

parpool(4);  % Create pool with 4 workers  
% Run inference or nested bootstrap  
delete(gcp('nocreate'));  % Close pool when finished  
      
Parallel computation can reduce runtime by 50-75% for embarrassingly parallel tasks like nested bootstrap.

### **4.5 Performance Evaluation**

**Note 21: Interpreting AUROC and AUPR.** AUROC (Area Under Receiver Operating Characteristic) and AUPR (Area Under Precision-Recall) measure different aspects of inference performance. AUROC is less sensitive to class imbalance and is useful for comparing methods across datasets. AUPR is more informative for highly sparse networks (which most GRNs are) because it focuses on the positive class (edges). A random predictor achieves AUROC \= 0.5 regardless of sparsity, but AUPR \= (number of edges) / (g²) for a random predictor, which is very low for sparse networks. Good inference methods typically achieve AUROC \> 0.7 and AUPR \> 0.3 on well-powered synthetic datasets (5).

**Note 22: Edge sign prediction.** Most inference methods can predict the presence of edges but predicting the correct sign (activation vs. repression) is more challenging. GeneSPIDER2 CompareModels can evaluate sign prediction accuracy separately. For applications where regulatory direction matters (e.g., drug target prediction), sign-aware metrics should be prioritized.

**Note 23: Biological validation.** High computational performance metrics do not guarantee biological relevance. Inferred networks should be validated through: (i) literature review of known interactions, (ii) enrichment analysis of functional categories, (iii) comparison with orthogonal data types (e.g., ChIP-seq, protein-protein interactions), and (iv) experimental validation of predicted novel edges. GeneSPIDER2 provides the computational framework, but biological interpretation requires domain expertise.

### **4.6 Troubleshooting**

**Note 24: Memory errors.** If MATLAB runs out of memory during network generation or inference, try: (i) closing other applications to free RAM, (ii) reducing network size or number of experiments, (iii) processing data in batches, or (iv) using a machine with more memory. For networks with thousands of genes, 32+ GB RAM is recommended.

**Note 25: Convergence issues in inference.** Some inference methods (particularly optimization-based methods like lasso) may fail to converge for ill-conditioned data. Solutions include: (i) checking data for numerical issues (infinite values, extreme outliers), (ii) adjusting method-specific parameters (e.g., convergence tolerance), (iii) trying alternative methods, or (iv) preprocessing data to improve conditioning (e.g., removing low-variance genes).

**Note 26: Unexpected performance.** If inference performance is much lower than expected on synthetic data, check: (i) that the SNR parameter is appropriate (not too high), (ii) that sufficient experiments/samples are provided (generally need at least g/2 perturbation experiments for reasonable performance), (iii) that the perturbation matrix P correctly reflects the experimental design, and (iv) that edge directionality is correctly interpreted (column to row in GeneSPIDER2).

**Note 27: Version compatibility.** GeneSPIDER2 is actively developed and updated. When using published results or shared code, note the GeneSPIDER2 version used. Different versions may produce slightly different results due to algorithm improvements or bug fixes. The version number can be checked in the repository or documentation.

### **4.7 Best Practices**

**Note 28: Documentation and reproducibility.** For reproducible research, document: (i) GeneSPIDER2 version, (ii) MATLAB version, (iii) all parameters used for network generation, data simulation, and inference, (iv) random seeds if applicable, and (v) complete processing pipeline. Save intermediate results (generated networks, simulated data, inferred networks) for later analysis or troubleshooting.

**Note 29: Benchmarking workflow.** A recommended workflow for benchmarking a new inference method is: (i) generate multiple synthetic networks with varying properties (size, sparsity, topology), (ii) simulate data with multiple noise levels and sample sizes, (iii) apply the new method and established baseline methods, (iv) compute performance metrics for all conditions, (v) identify conditions where the new method excels or struggles, and (vi) validate conclusions on real data. GeneSPIDER2 facilitates all steps of this workflow.

**Note 30: Data sharing and format conversion.** When sharing inferred networks, include: (i) the adjacency matrix in a standard format (.mat for MATLAB, .csv for universal access), (ii) gene names/identifiers, (iii) edge list format for easy import into other tools, (iv) metadata describing the inference method and parameters, and (v) performance metrics if the true network is known. The GRN Benchmark format provides a standardized structure for sharing and comparing results.

## **References**

1\. Huynh-Thu, V. A., Irrthum, A., Wehenkel, L. & Geurts, P. Inferring regulatory networks from expression data using tree-based methods. PLoS One 5, e12776 (2010).

2\. Stuart, T. & Satija, R. Integrative single-cell analysis. Nat. Rev. Genet. 20, 257-272 (2019).

3\. Dixit, A. et al. Perturb-Seq: Dissecting molecular circuits with scalable single-cell RNA profiling of pooled genetic screens. Cell 167, 1853-1866.e17 (2016).

4\. Datlinger, P. et al. Pooled CRISPR screening with single-cell transcriptome readout. Nat. Methods 14, 297-301 (2017).

5\. Tjärnberg, A., Morgan, D. C., Studham, M., Nordling, T. E. M. & Sonnhammer, E. L. L. GeneSPIDER \- gene regulatory network inference benchmarking with controlled network and data properties. Mol. Biosyst. 13, 1304-1312 (2017).

6\. Garbulowski, M., Hillerton, T., Morgan, D., Seçilmiş, D., Sonnhammer, L., Tjärnberg, A., Nordling, T. E. M. & Sonnhammer, E. L. L. GeneSPIDER2: large scale GRN simulation and benchmarking with perturbed single-cell data. NAR Genomics Bioinformatics 6, lqae121 (2024).

7\. Barabási, A. L. & Oltvai, Z. N. Network biology: understanding the cell's functional organization. Nat. Rev. Genet. 5, 101-113 (2004).

8\. Barabási, A. L. & Albert, R. Emergence of scaling in random networks. Science 286, 509-512 (1999).

9\. Morgan, D., Tjärnberg, A., Nordling, T. E. M. & Sonnhammer, E. L. L. A generalized framework for controlling FDR in gene regulatory network inference. Bioinformatics 35, 1026-1032 (2019).

10\. Nordling, T. E. M. & Jacobsen, E. W. On the estimation of ordinary differential equation parameters from noisy observations. IFAC Proceedings Volumes 45, 548-553 (2012).

11\. Han, H. et al. TRRUST v2: an expanded reference database of human and mouse transcriptional regulatory interactions. Nucleic Acids Res. 46, D380-D386 (2018).

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFgAAAANBAMAAADf+LRDAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAEM3v3burMomZRCJ2VGaYzhOJAAAAGElEQVR4XmP8z0A8YEIXwAdGFSODYa8YAGXUARkrzVgHAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAMBAMAAADFQ2OWAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAid3vzburmXZEVBBmMiJm6649AAAAFklEQVR4XmP8z4AbMKELIINRSQYCkgBrYAEXd7eMCQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKgAAAAkBAMAAADvF2wkAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAImaZu83vMlSrRHaJ3RAH4gJaAAAAMUlEQVR4Xu3MIQ4AIBDAMOD/fz48uoZklRPbs7zzBqGp19Rr6jX1mnpNvaZeU++f6QWiFAFHxMeF+wAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFAAAAAMBAMAAAAHcycSAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMARN3vzburIjKJdmaZEFTQbHVkAAAAGElEQVR4XmP8z0AcYEIXwAVGFeIFA6gQAAIvARfXByyPAAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAKgAAAAQBAMAAABw/CmfAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMARLvvMquJIpnNVBB2Zt2EZ/YjAAAAIUlEQVR4XmP8z0B9wIQuQA0waij1waih1AejhlIfDB1DAaRMAR958bOTAAAAAElFTkSuQmCC>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIQAAAANBAMAAACa1duEAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMARN3vzburIjKJdmaZEFTQbHVkAAAAHElEQVR4XmP8z0ApYEIXIB2MGoEAo0YgwLAxAgCSaQEZGCxJ1gAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAI0AAAAkBAMAAABGY6SwAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMAid3vzburmXZEVBBmMiJm6649AAAANElEQVR4Xu3MoREAMAjAQGD/ZTtB8RURHKIiLyOSNzacesuQH+aH+WF+mB/mh/lhfthvnwbR9gI4mA28OAAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAH8AAAANBAMAAAByeazqAAAAMFBMVEX///8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAv3aB7AAAAD3RSTlMARN3vzburIjKJdmaZEFTQbHVkAAAAHElEQVR4XmP8z0AR+MiELkIqGDVg1AAQGAYGAABWgQIKuA8M2wAAAABJRU5ErkJggg==>