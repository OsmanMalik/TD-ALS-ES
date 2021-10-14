function lin_samples = to_linear_idx_CP(samples, n, sz)
%to_linear_idx_CP Computes linear samples corresponding to subindices for
%CP design matrix
%
%lin_samples = to_linear_idx_CP(samples, n, sz) outputs the linear samples
%correponding to the input samples produced by draw_samples_CP. n should be
%the factor matrix being solved for, and sz should be a vector containing
%the number of rows of all factor matrices.

N = length(sz);
vec = @(x) x(:);
u = [1:n-1, n+1:N]; % Order vector
size_vector = sz(u);
size_vector = [1; vec(size_vector(1:end-1))];
lin_samples = (samples(:, u) - [0 ones(1,N-2)])*cumprod(size_vector);

end
