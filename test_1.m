% Test 1
%
% This script computes both the exact leverage score distribution and the
% leverage score distribution our proposed CP-ALS-ES uses. Moreover, in
% order to verify that the sampling scheme does what it is supposed to do,
% the script additionally draws a number of samples using the scheme, and
% then computes and plots the empirical distribution. The goal here is that
% all the distributions should roughly match.

X = data_loader('cat-12d');
X = X(1:3,1:4,1:5,:,1,1);
N = length(size(X));
n = 4;
R = 7;
J = 1000;
no_samp = 100000;
v = [N:-1:n+1, n-1:-1:1];

% CP Decomposition (to get realistic least squares problem)
CPD = cp_als(tensor(X), R, 'tol', 1e-5);

% Exact
mat = khatrirao(CPD.U(v));
r = rank(mat);
[Q, ~, ~] = svd(mat, 'econ');
Q = Q(:,1:r);
p_exact = sum(Q.^2, 2) / r;

% Proposal
PsiA = recursive_sketch_CP(CPD.U, n, J);
r = rank(PsiA);
[~, Sigma, V] = svd(PsiA, 'econ');
V = V(:,1:r);
Sigma = Sigma(1:r, 1:r);
ell = sum((mat * V * inv(Sigma)).^2, 2);
p_our = ell/sum(ell);

% Sampled distribution
samples = draw_samples_CP(CPD.U, PsiA, n, no_samp);
sz = cellfun(@(x) size(x,1), CPD.U);
lin_samples = to_linear_idx_CP(samples, n, sz);
occ = zeros(size(p_our));
for k = 1:length(lin_samples)
    occ(lin_samples(k)) = occ(lin_samples(k))+1;
end
p_sampled = occ/sum(occ);

% Plot distributions
figure
plot(p_exact, 'linewidth', 2)
hold on
plot(p_our, 'linewidth', 2)
plot(p_sampled, 'linewidth', 2)
legend('Exact', 'Our', 'Sampled')
axis tight
