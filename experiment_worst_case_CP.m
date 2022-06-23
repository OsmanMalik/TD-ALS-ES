% The purpose of this script is to provide an example when CP-ALS-ES is
% clearly superior to CP-ARLS-LEV along the lines of what the worst-case
% analysis indicates.
%
% This example uses randomized range finder (RRF) initialization, which is
% discussed e.g. in Appendix F of [LK20]. The initialization is done
% outside of the decomposition functions in order to better understand how
% much time the decompositions take without considering initialization.
%
%REFERENCES:
%
%   [LK20] B. W. Larsen, T. G. Kolda. "Practical Leverage-Based Sampling
%          for Low-Rank Tensor Decomposition". arXiv:2006.16438v3. 2020.

%% Setup
% Good settings for this example:
% N = 10;
% CP_rank = 4;
% I = 6;
% corner_val = 4;
% maxiters = 20;

% Set random seed for reproducibility
rng(1)

vec = @(x) x(:);

% Settings
N = 10;
CP_rank = 4;
I = 6;
corner_val = 4;
maxiters = 20;
A = cell(1,N);

% Create tensor
for n = 1:N
    A{n} = randn(I,CP_rank);
    A{n}(2:end,1) = 0;
    A{n}(1,2:end) = 0;
    A{n}(1,1) = corner_val;
end
X = double(tensor(ktensor(A)));
X = X + 1e-2*randn(size(X));
normX = norm(vec(X));

% Compute RRF initialization
my_tic = tic;
A_init = cell(1,N);
for n = 2:N
    Xn = classical_mode_unfolding(X,n);
    A_init{n} = Xn * randn(size(Xn,2), CP_rank);
end
toc_init = toc(my_tic);

fprintf('Factor matrix initialization:\n')
fprintf('\tTime: %.4f\n\n', toc_init)


%% Test of CP-ARLS-LEV

% Sketch size
J = I^(N-2); % Requires I^(N-2) to work properly; less gives lower accuracy
             % or numerical issues

% Run decomposition
my_tic = tic;
factor_mats = cp_arls_lev(X, CP_rank, J, 'maxiters', maxiters, 'A_init', A_init);
toc_cp_arls_lev = toc(my_tic);
lambda = ones(CP_rank,1);

% Compute error
Y = double(tensor(ktensor(lambda, factor_mats)));
er_cp_arls_lev = norm(vec(Y-X))/normX;

% Print results
fprintf('CP-ARLS-LEV\n')
fprintf('\tTime: %.4f s\n', toc_cp_arls_lev)
fprintf('\tError: %.4e\n\n', er_cp_arls_lev)


%% Test of CP-ALS-ES

% Sketch sizes
J1 = 1000;
J2 = 50;

% Run decomposition
my_tic = tic;
factor_mats = cp_als_es(X, CP_rank, J1, J2, 'maxiters', maxiters, 'A_init', A_init);
toc_cp_als_es = toc(my_tic);
lambda = ones(CP_rank,1);

% Compute error
Y = double(tensor(ktensor(lambda, factor_mats)));
er_cp_als_es = norm(vec(Y-X))/normX;

% Print results
fprintf('CP-ALS-ES\n')
fprintf('\tTime: %.4f s\n', toc_cp_als_es)
fprintf('\tError: %.4e\n\n', er_cp_als_es)

%% Test of CP-ALS

% Run decomposition
my_tic = tic;
M = cp_als(tensor(X), CP_rank, 'maxiters', maxiters, 'tol', 0, 'init', A_init);
toc_cp_als = toc(my_tic);

% Compute error
Y = double(tensor(ktensor(M)));
er_cp_als = norm(vec(Y-X))/normX;

% Print results
fprintf('CP-ALS\n')
fprintf('\tTime: %.4f s\n', toc_cp_als)
fprintf('\tError: %.4e\n\n', er_cp_als)
