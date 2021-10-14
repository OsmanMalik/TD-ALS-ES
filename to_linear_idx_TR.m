function lin_samples = to_linear_idx_TR(samples, n, sz)
%to_linear_idx_TR Computes linear samples corresponding to subindices for
%TR design matrix
%
%lin_samples = to_linear_idx_TR(samples, n, sz) outputs the linear samples
%corresponding to the input samples produced by draw_samples_TR. n should
%be the factor matrix being solved for, and sz should be a vector
%containing the size of the middle index of all cores.

N = length(sz);
vec = @(x) x(:);
q = [n+1:N, 1:n-1]; % Order vector
size_vector = sz(q);
size_vector = [1; vec(size_vector(1:end-1))];
lin_samples = (samples(:, q) - [0 ones(1,N-2)])*cumprod(size_vector);

end
