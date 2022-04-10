function A = cp_als_es(X, R, J1, J2, varargin)
%cp_als_es Computes CP decomposition via efficiently sampled ALS
%
%A = cp_als_es(X, R, J1, J2) returns the factor matrices in a cell A for a
%rank R CP decomposition of X. J1 and J2 are the sketch rates used for
%leverage score estimation and least squares sampling, respectively. See 
%our paper for details. 
%
%A = cp_als_es(___, 'maxiters', maxiters) can be used to control the
%maximum number of iterations. maxiters is 50 by default.
%
%A = cp_als_es(___, 'permute_for_speed', permute_for_speed) can be used to
%permute the modes of the input tensor so that the largest mode comes
%first. This can speed up the sampling, since all 1st indices can be drawn
%together rather than one at a time. Set permute_for_speed to true to
%enabler this. It is false by default.
%
%A = cp_als_es(___, 'A_init', A_init) can be used to set how the factor
%matrices are initialized. If A_init is "rand", then all the factor
%matrices are initialized to have entries drawn uniformly at random from
%[0,1]. If A_init is "RRF", then the factor matrices are initalized via a
%randomized range finder applied to the unfoldings of X. A_init can also be
%a cell array containing initializations for the factor matrices.
%
%This function currently does not support any checking of convergence
%criteria.

% Handle optional inputs
params = inputParser;
addParameter(params, 'maxiters', 50, @(x) isscalar(x) & x > 0);
addParameter(params, 'permute_for_speed', false);
addParameter(params, 'A_init', "rand")
parse(params, varargin{:});
maxiters = params.Results.maxiters;
permute_for_speed = params.Results.permute_for_speed;
A_init = params.Results.A_init;

N = ndims(X);

if isscalar(J1)
    J1 = repmat(J1, N, 1);
end
if isscalar(J2)
    J2 = repmat(J2, N, 1);
end

% Permute modes of X for increased speed
if permute_for_speed
    sz = size(X);
    [~, max_idx] = max(sz);
    perm_vec = mod((max_idx : max_idx+N-1) - 1, N) + 1;
    inv_perm_vec(perm_vec) = 1:N;
    X = permute(X, perm_vec);
    J1 = J1(perm_vec);
    J2 = J2(perm_vec);
end

sz = size(X);

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

% Main loop
for it = 1:maxiters
    
    % Iterate through all factor matrices
    for n = 1:N
        
        % Draw samples
        PsiA = recursive_sketch_CP(A, n, J1(n));
        [samples, sqrt_p] = draw_samples_CP(A, PsiA, n, J2(n));
        
        % Merge identical samples and count occurences
        [occurs, ~] = groupcounts(samples);
        [unq_samples, unq_idx] = unique(samples(:,[1:n-1 n+1:N]), 'rows');  % Since groupcounts can't output this...
        J2_unq = size(unq_samples,1);
        unq_samples = [unq_samples(:, 1:n-1) nan(J2_unq,1) unq_samples(:, n:N-1)];
        
        % Compute rescaling factors
        rescale = sqrt(occurs./J2(n)) ./ sqrt_p(unq_idx);
        
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
        
    end
    
    fprintf('\tIteration %d complete\n', it);
end

if permute_for_speed
    A = A(inv_perm_vec);
end

end
