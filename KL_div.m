function div = KL_div(p,q)
%KL_div Compute KL-divergence between two distributions p and q

div = sum(p.*log2(p./q));

end
