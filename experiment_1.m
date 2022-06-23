% Experiment 1
%
% This script computes the KL divergence between the exact leverage score
% distribution and the sampling distributions used by CP-ARLS-LEV and our
% CP-ALS-ES.

X = data_loader('cat-12d');
N = length(size(X));
n = N;
no_trial = 10;
Rs = [10 20];
Js = [1e+4 1e+3 1e+2];
v = [N:-1:n+1, n-1:-1:1];
results_table_KL_mean = nan(length(Js)+1, length(Rs));
results_table_KL_std = nan(length(Js)+1, length(Rs));
results_table_time = nan(length(Js)+1, length(Rs));
rng(1)

for rnk = 1:length(Rs)
    % CP Decomposition
    R = Rs(rnk);
    CPD = cp_als(tensor(X), R, 'tol', 1e-5);

    % Exact
    mat = khatrirao(CPD.U(v));
    r = rank(mat);
    [Q, ~, ~] = svd(mat, 'econ');
    Q = Q(:,1:r);
    p_exact = sum(Q.^2, 2) / r;

    % CP-ARLS-LEV
    my_tic = tic;
    p_sub = cell(1,N-1);
    for k = 1:N-1
        r = rank(CPD.U{v(k)});
        [Q, ~, ~] = svd(CPD.U{v(k)}, 'econ');
        Q = Q(:,1:r);
        p_sub{k} = sum(Q.^2, 2) / r;
    end
    p_LK = khatrirao(p_sub);
    my_toc = toc(my_tic);
    KL_div_LK = KL_div(p_LK, p_exact);
    results_table_KL_mean(1, rnk) = KL_div_LK;
    results_table_time(1, rnk) = my_toc;
    
    % CP-ALS-ES (our proposal)
    for j = 1:length(Js)
        J1 = Js(j);
        KL_div_our = zeros(1,no_trial);
        my_toc_total = 0;
        for tr = 1:no_trial
            my_tic = tic;
            PsiA = recursive_sketch_CP(CPD.U, N, J1);
            r = rank(PsiA);
            [~, Sigma, V] = svd(PsiA, 'econ');
            V = V(:,1:r);
            Sigma = Sigma(1:r, 1:r);
            ell = sum((mat * V * inv(Sigma)).^2, 2);
            p_our = ell/sum(ell);
            my_toc = toc(my_tic);
            my_toc_total = my_toc_total + my_toc;
            KL_div_our(tr) = KL_div(p_our, p_exact);
        end
        results_table_KL_mean(j+1, rnk) = mean(KL_div_our);
        results_table_KL_std(j+1, rnk) = std(KL_div_our);
        results_table_time(j+1, rnk) = my_toc_total/no_trial;
    end
end

results_table_KL_mean
results_table_KL_std
results_table_time
