// neg binom parameterization
// estimate correlation matrix among cell types

// other bucket
// test first with other proportion of 0. i.e. with synthetic mixtures of known cell types
// then simulate other data:
// either pick other genes that we call tumor-related and give some values for those and spike in
// or spike in cell line or RCC sample (though that may have immune content)

data {
    // dimensions
    int<lower=1> N;  // N obs
    int<lower=1> G;  // N genes
    int<lower=1> S;  // N samples
    int<lower=0> C;  // N classes (e.g. B-cell, T-cell, B_Naive, CD5, CD45RO, etc)
                     //     note: classes should be mutually exclusive. Each row here should sum to 1
    int<lower=0> M; // number of cell-level predictors 
   
    // data for each gene*sample
    int<lower=1, upper=G> gene[N];    // gene id for each obs
    int<lower=1, upper=S> sample[N];  // sample id for each obs
    vector<lower=0, upper=1>[C] x[N]; // map each obs to each class (0:'- or ?', 1:'+')
    int<lower=0> y[N];                // count/tpm for each obs
    
    // group-level predictors for each class C
    matrix[C, M] cell_features; 

    // out of sample estimates, with unknown comp
    int<lower=1> N2;           // number of records in out of sample (UNK)
    int<lower=1> S2;           // number of samples in UNK set
    int<lower=1, upper=G> gene2[N2];    // gene id for UNK data (corresponding to IDs above)
    int<lower=1, upper=S2> sample2[N2]; // sample id for each UNK sample (separate from above)
    int<lower=0> y2[N2];       // data for UNK set 
}
transformed data {
    int sample_y[S, G];    // array (size SxG) of ints
    vector[C] sample_x[S]; // array (size S) of vectors[C]
    int sample2_y[S2, G];
    int<lower=1> nu;
    for (n in 1:N) {
        sample_y[sample[n], gene[n]] = y[n];
        sample_x[sample[n]] = x[n,];
    }
    for (n in 1:N2) {
        sample2_y[sample2[n], gene2[n]] = y2[n];
    }
    nu = 1;
}
parameters {
    cholesky_factor_corr[C] Omega_L;
    vector<lower=0>[C] Omega_sigma;
    //corr_matrix[C] Omega;        // degree of correlation among loading factors for each cell type
    //vector<lower=0>[C] tau;      // scale for each cell type - multiplied (on diagonal) with Omega
    
    matrix<lower=0>[G, C] theta; // loading factors for each gene, for each cell type
    vector[C] theta_mu;          // mean expression level for each cell type
    vector[M] theta_coefs_raw;
    vector[M] theta_coefs_per_gene[G];
    
    vector[G] log_gene_base;     // constant intercept expression level for each gene, irrespective of cell type
    vector<lower=0>[G] gene_phi; // overdispersion parameter per transcript (for now)
    simplex[C] sample2_x[S2];     // inferred sample2 compositions (simplex type enforces sum-to-one)

    vector<lower=0, upper=1>[S2] unknown_prop; // proportion of each test sample that is of unknown cell type
    vector[G] other_log_contribution_per_gene[S2]; // for each test sample, per-transcript contribution of unknown cell type
}
transformed parameters {
    vector[M] theta_coefs[G];
    matrix[C, C] Omega_sigma_L_multiplied;
    vector[C] theta_tmp[G]; // temporary predictor for cell-gene-specific expression level
    vector[G] log_expected_rate[S];
    vector[G] log_expected_rate2[S2];

    for (g in 1:G) {
        theta_coefs[g] = theta_coefs_raw + theta_coefs_per_gene[g];
        theta_tmp[g] = theta_mu + cell_features*theta_coefs[g];
    }

    for (s in 1:S)
        log_expected_rate[s] = log_gene_base + log(theta*sample_x[s]);

    for (s in 1:S2)
        log_expected_rate2[s] = log_gene_base + log(theta*sample2_x[s]) * (1 - unknown_prop[s]) + other_log_contribution_per_gene[s] * unknown_prop[s];

    Omega_sigma_L_multiplied = diag_pre_multiply(Omega_sigma, Omega_L);
}
model {
    // estimate theta - gene-level expression per cell type, as a function of cell-surface expression proteins
    theta_mu ~ normal(0, 1);
    theta_coefs_raw ~ normal(0, 1);
    Omega_sigma ~ gamma(0.1, 0.1);
    Omega_L ~ lkj_corr_cholesky(nu);
    for (g in 1:G) {
        theta_coefs_per_gene[g] ~ normal(0, 1);
        theta[g] ~ multi_normal_cholesky(theta_tmp, Omega_sigma_L_multiplied);
    }

    // estimate sample_y: observed expression for a sample (possibly a mixture)
    log_gene_base ~ normal(0, 1);
    gene_phi ~ normal(0, 1);
    for (s in 1:S) {        
        sample_y[s] ~ neg_binomial_2_log(log_expected_rate[s], gene_phi);
    }
    
    // estimate sample2_y: observed expression for a sample of unknown composition
    unknown_prop ~ beta(5, 5); // not sure about this, maybe Beta(1,1) uniform?

    for (s in 1:S2) {
        other_log_contribution_per_gene[s] ~ normal(0, 1);
        sample2_y[s] ~ neg_binomial_2_log(log_expected_rate2[s], gene_phi);
    }
}
generated quantities {
    int y_rep[N];
    real log_lik[N];
    matrix[C,C] Omega;
    matrix[C,C] tau;
    Omega = multiply_lower_tri_self_transpose(Omega_L);
    tau = quad_form_diag(Omega_L, Omega_sigma);
    
    for (n in 1:N) {
        real log_expected_rate_rep;
        log_expected_rate_rep = log_gene_base[gene[n]] + log(theta[gene[n], ]*x[n]);
        y_rep[n] = neg_binomial_2_log_rng(log_expected_rate_rep, gene_phi[gene[n]]);
        log_lik[n] = neg_binomial_2_log_lpmf(y[n] | log_expected_rate_rep, gene_phi[gene[n]]);
    }
}