function A = cp_arls_lev(X, R, J, varargin)
%cp_arls_lev Simplified implementation of CP-ARLS-LEV by [LK20]
%
%A = cp_arls_lev(X, R, J) returns the factor matrices in a cell A for a
%rank R CP decomposition of X. The decompostion is computed using
%CP-ARLS-LEV proposed in [LK20], specifically Alg. 6.3 in that paper. The
%implementation here is simplified, missing some of the bells and whistles
%of the full algorithm. In particular, there is no hybrid-deterministic
%sampling, but repeated rows are combined. There is currently also no
%checking of convergence criteria implemented.
%
%A = cp_arls_lev(___, 'maxiters', maxiters) can be used to control the
%maximum number of iterations. maxiters is 50 by default.
%
%REFERENCES:
%
%   [LK20] B. W. Larsen, T. G. Kolda. "Practical Leverage-Based Sampling
%          for Low-Rank Tensor Decomposition". arXiv:2006.16438. 2020.

% Handle optional inputs
params = inputParser;
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
parse(params, varargin{:});
maxiters = params.Results.maxiters;

sz = size(X);
N = length(sz);

% Initialize factor matrices
A = cell(1,N);
for j = 2:N
    A{j} = randn(sz(j), R);
end

% Initialize sampling probability
sampling_probs = cell(1,N);
for j = 2:N
    U = col(A{j});
    sampling_probs{j} = sum(U.^2, 2) / size(U, 2);
end

% Main loop
for it = 1:maxiters
    
    % Iterate through all factor matrices
    for n = 1:N
        
        % Draw samples
        samples = nan(J, N);
        for j = 1:N
            if j ~= n
                samples(:, j) = randsample(sz(j), J, true, sampling_probs{j});
            end
        end

        % Merge identical samples and count occurences
        [occurs, unq_samples_cell] = groupcounts(samples);
        J_unq = length(occurs);
        unq_samples = nan(J_unq, N);
        for j = 1:N
            unq_samples(:, j) = unq_samples_cell{j};
        end
        
        % Compute rescaling factors
        rescale = sqrt(occurs./J);
        for j = 1:N
            if j ~= n
                rescale = rescale ./ sqrt(sampling_probs{j}(unq_samples(:,j)));
            end
        end
        
        % Construct sketched design matrix
        SA = repmat(rescale, 1, R);
        for j = N:-1:1
            if j ~= n
                SA = SA .* A{j}(unq_samples(:, j), :);
            end
        end
        
        % Construct sketched right hand side
        lin_samples = to_linear_idx_CP(unq_samples, n, sz);
        Xn = classical_mode_unfolding(X, n);
        SXnT = Xn(:, lin_samples).';
        SXnT = SXnT .* rescale;
        
        % Solve sketched LS problem and update nth factor matrix
        A{n} = (SA \ SXnT).';
        
        % Update nth sampling distribution
        U = col(A{n});
        sampling_probs{n} = sum(U.^2, 2) / size(U, 2);
    end
    
    fprintf('\tIteration %d complete\n', it);
end

end
