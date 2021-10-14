% Test 2
%
% This script computes both the exact leverage score distribution and the
% leverage score distribution our proposed TR-ALS-ES uses. Moreover, in
% order to verify that the sampling scheme does what it is supposed to do,
% the script additionally draws a number of samples using the scheme, and
% then computes and plots the empirical distribution. The goal here is that
% all the distributions should roughly match.
%
%Note that this function requires some files from the tr-als-sampled
%repo which is available at: https://github.com/OsmanMalik/tr-als-sampled

X = data_loader('cat-12d');
X = X(1:3,1:4,1:5,:,1,1);
N = length(size(X));
n = 4;
R = 4;
J = 1000;
no_samp = 100000;

% TR Decomposition (to get realistic least squares problem)
cores = tr_als(X, [2 3 4 5], 'verbose', true);
cores = cores';
G = cores(1:end-1);

% Exact
mat = subchain_matrix(cores, n);
r = rank(mat);
[Q, ~, ~] = svd(mat, 'econ');
Q = Q(:,1:r);
p_exact = sum(Q.^2, 2) / r;

% Proposal
PsiG_2 = recursive_sketch_TR(cores, n, J);
r = rank(PsiG_2);
[~, Sigma, V] = svd(PsiG_2, 'econ');
V = V(:,1:r);
Sigma = Sigma(1:r, 1:r);
ell = sum((mat * V * inv(Sigma)).^2, 2);
p_our = ell/sum(ell);

% Sampled distribution
samples = draw_samples_TR(cores, PsiG_2, n, no_samp);
sz = cellfun(@(x) size(x,2), cores);
lin_samples = to_linear_idx_TR(samples, n, sz);
occ = zeros(size(p_our));
for k = 1:length(lin_samples)
    occ(lin_samples(k)) = occ(lin_samples(k))+1;
end
p_sampled = occ/sum(occ);

% Plot distributions
figure
plot(p_exact, 'linewidth', 2)
hold on
plot(p_our, '--', 'linewidth', 2)
plot(p_sampled, 'linewidth', 2)
legend('Exact', 'Our', 'Sampled')
axis tight
