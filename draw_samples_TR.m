function [samples, sqrt_probs] = draw_samples_TR(G, sketch, n, J2)
%draw_samples_TR Draws samples for TR design matrix as discussed in our
%paper
%
%samples = draw_samples_TR(G, sketch, n, no_samp) returns no_samp samples
%organized into an no_samp by N matrix, where N is the number of modes of
%the tensor being decomposed. The cell G should contain all the TR core
%tensors in standard order, and the n-th core is the one being solved for.
%The n-th column of samples will just be NaN, since that index is not being
%sampled.
%
%Note that this function requires some files from the tr-als-sampled
%repo which is available at: https://github.com/OsmanMalik/tr-als-sampled

N = length(G);

% Compute matrix Phi
[~, Sigma, V] = svd(sketch, 'econ');
temp = V*inv(Sigma);
Phi = temp*temp';

% Precompute the two terms in Eq (23) in paper, but reshaped into tensors
% according to the discussion in Remark 19 of the paper
C1 = cell(1,N);
C1_unf = cell(1,N);
C2 = cell(1,N);
C2_vec = cell(1,N);
for j = 1:N
    [R0, I1, R1] = size(G{j});
    if j ~= n
        % Appropriate tensorization of first term
        G_2 = mode_unfolding(G{j}, 2);
        temp = khatrirao(G_2.', G_2.');
        temp = reshape(temp, R1, R0, R1, R0, I1);
        temp = permute(temp, [2 4 5 1 3]);
        C1{j} = reshape(temp, R0^2, I1, R1^2);
        C1_unf{j} = classical_mode_unfolding(C1{j}, 2);
        
        % Appropriate tensorization of second term
        temp = G_2.' * G_2;
        temp = reshape(temp, R1, R0, R1, R0);
        temp = permute(temp, [2 4 1 3]);
        C2{j} = reshape(temp, R0^2, R1^2);
        C2_vec{j} = C2{j}(:).';
        
    else
        % Also store permuted Phi in with (G_[2]^(j))^T G_[2]^(j) matrices
        temp = reshape(Phi, R0, R1, R0, R1);
        temp = permute(temp, [1 3 2 4]);
        C2{j} = reshape(temp, R0^2, R1^2);
        
    end
end

% Precompute vectors with order in which to multiply cores.
% For some reason, reinitializing the vector below ends up taking some
% non-negligible amount of time if it's done inside the main loop below, so
% predefining them here instead to avoid that.
idx_order = cell(N,1);
for m = 1:N
    idx_order{m} = [m+1:N 1:m-1];
end

% Main loop for drawing all samples
samples = nan(J2, N); % Each row is a realization (i_j)_{j \neq n}
sqrt_probs = ones(J2, 1); % To store the square root of the probability of each drawn sample
if n == 1
    first_idx = 2;
else
    first_idx = 1;
end
first_idx_flag = true;


for samp = 1:J2
    % Compute P(i_m | (i_j)_{j < m, ~= n}) for each m (~=n) and draw i_m
    for m = first_idx:N
        if m == first_idx && ~first_idx_flag
            continue
        end
        
        if m ~= n
            
            % Compute conditional probability vector
            idx = idx_order{m};
            if idx(1) >= m || idx(1) == n
                M = C2{idx(1)};
            else
                sz = size(C1{idx(1)}, 1:3);
                M = reshape(C1{idx(1)}(:, samples(samp,idx(1)), :), sz(1), sz(3));
            end
            for j = 2:length(idx)
                if idx(j) >= m || idx(j) == n
                    M = M * C2{idx(j)};
                else
                    sz = size(C1{idx(j)}, 1:3);
                    M = M * reshape(C1{idx(j)}(:, samples(samp,idx(j)), :), sz(1), sz(3));
                end
            end
            common_terms = M';
            
            common_terms_vec = common_terms(:);
            %const = C2_vec{m} * common_terms_vec;
            %common_terms_vec = common_terms_vec / const;
            prob_m = C1_unf{m} * common_terms_vec;
            prob_m = prob_m / sum(prob_m);      

            if first_idx_flag
                % Draw from the vector
                if min(prob_m)<0
                    prob_m = max(prob_m, 0);
                end
                if sum(isnan(prob_m))>0
                    error("Probability vector contains NaN")
                end
                samples(:, m) = randsample(length(prob_m), J2, true, prob_m);

                % Update probability vector
                sqrt_probs = sqrt(prob_m(samples(:, m)));
                
                first_idx_flag = false;
            else
                % Draw from the vector
                if min(prob_m)<0
                    prob_m = max(prob_m, 0);
                end
                if sum(isnan(prob_m))>0
                    error("Probability vector contains NaN")
                end
                samples(samp, m) = mt19937ar(prob_m);
                
                % Update probability vector
                sqrt_probs(samp) = sqrt_probs(samp) * sqrt(prob_m(samples(samp,m)));
            end
        end
    end
end

end
