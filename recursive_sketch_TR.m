function sketch = recursive_sketch_TR(G, n, J, varargin)
%recursive_sketch_TR Computes recursive sketch of TR design matrix
%
%sketch = recursive_sketch_TR(G, J) outputs the recursive sketch of the
%TR-ALS design matrix corresponding to updating the n-th core tensor. All
%the current core tensors (including the n-th) are given in the cell G.
%These should be the 1st, 2nd, etc, core tensors. The target embedding
%dimension is J.
%
%Example: When decomposing a 3-way tensor, G should contain the three core
%tensors. If we're updating the 2nd core tensor, then the call would be
%recursive_sketch_TR(G, 2, J).
%
%Note that this function requires some files from the tr-als-sampled
%repo which is available at: https://github.com/OsmanMalik/tr-als-sampled
%
%The recursive sketch was proposed by [AKK+20].
%
%REFERENCES:
%
%   [AKK+20] Ahle et al. Oblivious sketching of high-degree polynomial
%   kernels. SODA, p 141-160, 2020.

% Handle optional inputs
params = inputParser;
addParameter(params, 'verbose', false);
parse(params, varargin{:});
verbose = params.Results.verbose;

N = length(G);
q = ceil(log2(N-1));
Y = cell(1,N-1);

w = [n-1:-1:1 N:-1:n+1]; % Order vector, also called w in the paper

% Compute initial Y_j^0 for each 1 <= n <= N-1
for j = 1:N-1
    [R0, I1, R1] = size(G{w(j)});
    G_2 = classical_mode_unfolding(G{w(j)}, 2);
    idx_map = randsample(J, I1, true);
    sgn_map = round(rand(I1, 1))*2-1;
    CG_2 = countSketch(G_2', int64(idx_map), J, sgn_map, true)';
    Y{j} = classical_mode_folding(CG_2, 2, [R0 J R1]);
end

% Compute Y_j^m recursively for m = 1, ..., q
for m = 1:q
    Y_old = Y;
    Y = cell(1, ceil(length(Y_old)/2));
    
    for j = 1:length(Y)
        
        %  Draw hash functions that TensorSketch T_j^{(m)} will use
        h = cell(2,1);
        h{1} = randi(J, J, 1);
        h{2} = randi(J, J, 1);
        s = cell(2,1);
        s{1} = randi(2, J, 1)*2-3;
        s{2} = randi(2, J, 1)*2-3;
        
        % Combine tensors pairwise via TensorSketch
        if 2*j <= length(Y_old)
            Rj = size(Y_old{2*j-1}, 3);
            Rj_2 = size(Y_old{2*j}, 1);
            Y{j} = nan(Rj_2, J, Rj);
            if verbose
                fprintf("m = %d. Combining old tensors %d and %d into new tensor %d\n", m, 2*j-1, 2*j, j);
            end
            for rj_2 = 1:Rj_2
                for rj = 1:Rj
                    M1 = squeeze(Y_old{2*j-1}(:, :, rj))';
                    M2 = squeeze(Y_old{2*j}(rj_2, :, :));
                    M_out = sum(TensorSketch({M1, M2}, J, 'h', h, 's', s), 2); % Sum over "rj_1 = 1:Rj_1"
                    Y{j}(rj_2, :, rj) = M_out;
                end
            end
        else
            % The lines below are distributionally the same as doing a
            % TensorSketch with a vector with a single nonzero entry which
            % is +1 or -1
            [R0, ~, R1] = size(Y_old{2*j-1});
            Y_2 = classical_mode_unfolding(Y_old{2*j-1}, 2);
            CY_2 = countSketch(Y_2', int64(h{1}), J, s{1}, true)';
            Y{j} = classical_mode_folding(CY_2, 2, [R0 J R1]);
            if verbose
                fprintf("m = %d. CountSketching old tensor %d into new tensor %d\n", m, 2*j-1, j);
            end
        end
    end
end

sketch = mode_unfolding(Y{1}, 2);

end
