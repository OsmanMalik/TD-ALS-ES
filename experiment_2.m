% Experiment 2
%
% This script computes the KL divergence between the exact leverage score
% distribution and the sampling distributions used by TR-ALS-Sampled and
% our TR-ALS-ES.
%
% Note that this function requires some files from the tr-als-sampled
% repo which is available at: https://github.com/OsmanMalik/tr-als-sampled

X = data_loader('cat-12d');
N = length(size(X));
n = N;
no_trial = 10;
Rs = [3 5];
Js = [1e+4 1e+3 1e+2];
w = [n-1:-1:1 N:-1:n+1]; % Order vector, also called w in the paper
results_table_KL_mean = nan(length(Js)+1, length(Rs));
results_table_KL_std = nan(length(Js)+1, length(Rs));
results_table_time = nan(length(Js)+1, length(Rs));
rng(1)

for rnk = 1:length(Rs)
    % TR Decomposition
    R = Rs(rnk);
    cores = tr_als(X, R*ones(1,N), 'verbose', true);
    cores = cores';

    % Exact
    mat = subchain_matrix(cores, N);
    r = rank(mat);
    [Q, ~, ~] = svd(mat, 'econ');
    Q = Q(:,1:r);
    p_exact = sum(Q.^2, 2) / r;

    % TR-ALS-Sampled
    my_tic = tic;
    p_sub = cell(1,N-1);
    for k = 1:N-1
        G_2 = classical_mode_unfolding(cores{w(k)}, 2);
        r = rank(G_2);
        [Q, ~, ~] = svd(G_2, 'econ');
        Q = Q(:,1:r);
        p_sub{k} = sum(Q.^2, 2) / r;
    end
    p_MB = khatrirao(p_sub);
    my_toc = toc(my_tic);
    KL_div_MB = KL_div(p_MB, p_exact);
    results_table_KL_mean(1, rnk) = KL_div_MB;
    results_table_time(1, rnk) = my_toc;

    
    % TR-ALS-ES (our proposal)
    for j = 1:length(Js)
        J1 = Js(j);
        KL_div_our = zeros(1,no_trial);
        my_toc_total = 0;
        for tr = 1:no_trial
            my_tic = tic;
            PsiG_2 = recursive_sketch_TR(cores, N, J1);
            r = rank(PsiG_2);
            [~, Sigma, V] = svd(PsiG_2, 'econ');
            V = V(:,1:r);
            Sigma = Sigma(1:r, 1:r);
            ell = sum((mat * V * inv(Sigma)).^2, 2);
            p_our = ell/sum(ell);
            my_toc = toc(my_tic);
            my_toc_total = my_toc_total + my_toc;
            KL_div_our(tr) = KL_div(p_our, p_exact);
        end
        my_toc = toc(my_tic);
        results_table_KL_mean(j+1, rnk) = mean(KL_div_our);
        results_table_KL_std(j+1, rnk) = std(KL_div_our);
        results_table_time(j+1, rnk) = my_toc_total/no_trial;
    end
end

results_table_KL_mean
results_table_KL_std
results_table_time
