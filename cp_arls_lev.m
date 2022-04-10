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
%A = cp_arls_lev(___, 'A_init', A_init) can be used to set how the factor
%matrices are initialized. If A_init is "rand", then all the factor
%matrices are initialized to have entries drawn uniformly at random from
%[0,1]. If A_init is "RRF", then the factor matrices are initalized via a
%randomized range finder applied to the unfoldings of X. A_init can also be
%a cell array containing initializations for the factor matrices.
%
%REFERENCES:
%
%   [LK20] B. W. Larsen, T. G. Kolda. "Practical Leverage-Based Sampling
%          for Low-Rank Tensor Decomposition". arXiv:2006.16438. 2020.

% Handle optional inputs
params = inputParser;
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'A_init', "rand")
parse(params, varargin{:});
maxiters = params.Results.maxiters;
A_init = params.Results.A_init;

sz = size(X);
N = length(sz);

if isscalar(J)
    J = repmat(J, N, 1);
end

% Initialize factor matrices
if iscell(A_init)
    A = A_init;
else
    A = cell(1,N);
    for j = 2:N
        if strcmp(A_init, "rand")
        A{j} = rand(sz(j), R);
        elseif strcmp(A_init, "RRF")
            Xn = classical_mode_unfolding(X,j);
            A{j} = Xn * randn(size(Xn,2), R);
        end
    end
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
        samples = nan(J(n), N);
        for j = 1:N
            if j ~= n
                samples(:, j) = randsample(sz(j), J(n), true, sampling_probs{j});
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
        rescale = sqrt(occurs./J(n));
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
        
        % Note: Out of 2 options below, second option is MUCH faster (about
        % 30-40x in one test).

        % Option 1: Unfold matrix, compute columns to sample, then
        % transpose
        %{
        lin_samples = to_linear_idx_CP(unq_samples, n, sz);
        Xn = classical_mode_unfolding(X, n);
        SXnT = Xn(:, lin_samples).';
        %}

        % Option 2: Compute linear indices, sample directly from tensor,
        % then reshape to sketched matrix
        szp = cumprod([1 sz(1:end-1)]);
        samples_temp = unq_samples - 1; samples_temp(:,n) = 0;
        llin = 1+samples_temp*szp';
        llin = repelem(llin, sz(n), 1) + repmat((0:sz(n)-1)'*szp(n), size(unq_samples,1), 1);
        SXnT = X(llin);
        SXnT = reshape(SXnT, sz(n), size(unq_samples,1))';

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
