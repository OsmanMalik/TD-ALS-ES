% The purpose of this script is to provide an example when TR-ALS-ES is
% clearly superior to TR-ALS-Sampled along the lines of what the worst-case
% analysis indicates.
%
% This example uses randomized range finder (RRF) initialization. This is
% discussed for the CP decomposition in Appendix F of [LK20]. This idea is
% adapted in this script for the TR decomposition. The initialization is
% done outside of the decomposition functions in order to better understand
% how much time the decompositions take without considering initialization.
%
%REFERENCES:
%
%   [LK20] B. W. Larsen, T. G. Kolda. "Practical Leverage-Based Sampling
%          for Low-Rank Tensor Decomposition". arXiv:2006.16438v3. 2020.

% Set random seed for reproducibility
rng(1)

vec = @(x) x(:);

% Settings
N = 10;
TR_rank = 3*ones(1,N);
I = 6;
corner_val = 3;
maxiters = 20;
cores = cell(1,N);

% Create tensor
for n = 1:N
    cores{n} = zeros(TR_rank(n), I, TR_rank(mod(n,N)+1));
    cores{n}(1, 1, 1) = corner_val;

    %cores{n} = randn(TR_rank(n), I, TR_rank(mod(n,N)+1));
end
X = cores_2_tensor(cores);
X = X + 1e-2*randn(size(X));
normX = norm(vec(X));

% Compute RRF initialization
my_tic = tic;
cores_init = cell(1, N);
cores_init{1} = randn(TR_rank(1), I, TR_rank(2));
for n = 2:N
    Xn = classical_mode_unfolding(X, n);
    R0 = TR_rank(n);
    R1 = TR_rank(mod(n,N)+1); 
    XnS = Xn*randn(size(Xn,2), R0*R1);
    cores_init{n} = classical_mode_folding(XnS, 2, [R0 I R1]);
end
toc_init = toc(my_tic);

fprintf('Core tensor initialization:\n')
fprintf('\tTime: %.4f\n\n', toc_init)


%% Test of TR-ALS-ES

% Sketch sizes
J1 = 10000; % 10000
J2 = 1000; %100

% Run decomposition
my_tic = tic;
cores = tr_als_es(X, TR_rank, J1, J2*ones(1,N), ...
    'maxiters', maxiters, ...
    'tol', 0, ...
    'verbose', true, ...
    'init', cores_init);
toc_tr_als_es = toc(my_tic);

% Compute error
Y = cores_2_tensor(cores);
er_tr_als_es = norm(vec(X-Y))/normX;

% Print results
fprintf('TR-ALS-ES\n')
fprintf('\tTime: %.4f s\n', toc_tr_als_es)
fprintf('\tError: %.4e\n\n', er_tr_als_es)


%% Test of TR-ALS-Sampled

% Sketch size
J = I^(N-1)/2; % 2*I^(N-1) doesn't work; will need to make larger...
             % Note though that I^(N-1) is the number of rows in the full
             % LS problem.

% Run decomposition
my_tic = tic;
cores = tr_als_sampled(X, TR_rank, J*ones(1,ndims(X)), ...
    'tol', 0, ...
    'maxiters', maxiters, ...
    'alpha', 0, ...
    'verbose', true, ...
    'init', cores_init);
toc_tr_als_samp = toc(my_tic);

% Compute error
Y = cores_2_tensor(cores);
er_tr_als_samp = norm(vec(X-Y))/normX;

% Print results
fprintf('TR-ALS-Sampled\n')
fprintf('\tTime: %.4f s\n', toc_tr_als_samp)
fprintf('\tError: %.4e\n\n', er_tr_als_samp)



%% Test of TR-ALS

% Run decomposition
my_tic = tic;
cores = tr_als(X, TR_rank, ...
    'maxiters', maxiters, ...
    'tol', 0, ...
    'verbose', true, ...
    'init', cores_init);
toc_tr_als = toc(my_tic);

% Compute error
Y = cores_2_tensor(cores);
er_tr_als = norm(vec(X-Y))/normX;

% Print results
fprintf('TR-ALS\n')
fprintf('\tTime: %.4f s\n', toc_tr_als)
fprintf('\tError: %.4e\n\n', er_tr_als)
