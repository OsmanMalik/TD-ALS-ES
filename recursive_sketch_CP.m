function sketch = recursive_sketch_CP(A, n, J)
%recursive_sketch_CP Computes recursive sketch of CP design matrix
%
%sketch = recursive_sketch_CP(A, J) outputs the recursive sketch of the
%CP-ALS design matrix corresponding to updating the n-th factor matrix. All
%the current factor matrices (including the n-th) are given in the cell A.
%These should be the 1st, 2nd, etc, factor matrices. The target embedding
%dimension is J.
%
%Example: When decomposing a 3-way tensor, A should contain the three
%factor matrices. If we're updating the 2nd factor matrix, then the call
%would be recursive_sketch_CP(A, 2, J). 
%
%The recursive sketch was proposed by [AKK+20].
%
%REFERENCES:
%
%   [AKK+20] Ahle et al. Oblivious sketching of high-degree polynomial
%   kernels. SODA, p 141-160, 2020.

N = length(A);
R = size(A{2},2);
q = ceil(log2(N-1));
Y = cell(1,2^q);
v = [N:-1:n+1 n-1:-1:1]; % Order vector, also called v in our paper
sz = cellfun(@(x) size(x,1), A(v));

% Compute Y_j^0 for each 1 <= n <= N-1
for j = 1:N-1
    idx_map = randsample(J, sz(j), true);
    sgn_map = round(rand(sz(j), 1))*2-1;
    Y{j} = countSketch(A{v(j)}', int64(idx_map), J, sgn_map, true)';
end

% Compute Y_j^0 for each N <= n <= 2^q
for j = N:2^q
    Y{j} = zeros(J, R);
    idx = randsample(J, 1);
    sgn = round(rand())*2-1;
    Y{j}(idx, :) = sgn;
end

% Compute Y_j^m recursively for m = 1, ..., q
for m = 1:q
    for j = 1:2^(q-m)
        Y{j} = TensorSketch({Y{2*j-1}, Y{2*j}}, J);
    end
end

sketch = Y{1};

end
