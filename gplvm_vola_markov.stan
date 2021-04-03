functions{
    matrix cov_linear(vector[] X1, vector[] X2, real sigma){
        int N = size(X1);
        int M = size(X2);
        int Q = num_elements(X1[1]);
        matrix[N,M] K;
        {
            matrix[N,Q] x1;
            matrix[M,Q] x2;
            for (n in 1:N)
                x1[n,] = X1[n]';
            for (m in 1:M)
                x2[m,] = X2[m]';
            K = x1*x2';
        }
        return square(sigma)*K;}
    
    matrix cov_exp(vector[] X1, vector[] X2, real sigma, real l, real jitter){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        int Q = rows(X1[1]);
        for (n in 1:N)
            for (m in 1:M){
                dist = 0;  //sqrt(squared_distance(X1[n], X2[m]) + jitter);
                for (i in 1:Q)
                    dist = dist + fabs(X1[n,i] - X2[m,i]);
                K[n,m] = square(sigma) * exp(-0.5/l * dist);
            }
        return K;}
    
    matrix cov_matern32(vector[] X1, vector[] X2, real sigma, real l, real jitter){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        for (n in 1:N)
            for (m in 1:M){
                dist = sqrt(squared_distance(X1[n], X2[m]) + jitter);
                K[n,m] = square(sigma)*(1+sqrt(3)*dist/l)*exp(-sqrt(3)*dist/l);
            }
        return K;}
    
    matrix cov_matern52(vector[] X1, vector[] X2, real sigma, real l, real jitter){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        for (n in 1:N)
            for (m in 1:N){
                dist = sqrt(squared_distance(X1[n], X2[m]) + jitter);
                K[n,m] = square(sigma)*(1+sqrt(5)*dist/l+5*square(dist)/(3*square(l)))*exp(-sqrt(5)*dist/l);
            }
        return K;}
    
    matrix kernel_f(vector[] X1, vector[] X2, real kernel_l, vector Sigma, int kernel_number, vector noise_std, real jitter){
        int N = size(X1);
        matrix[N,N] K;
        
        if (kernel_number==1){
            K = cov_linear(X1, X2, kernel_l);
            K = quad_form_diag(K, Sigma);
            for (n in 1:N)
                K[n,n] = K[n,n] + pow(noise_std[n], 2) + jitter;}
                
        else if (kernel_number==2){
            K = cov_exp_quad(X1, X2, 1., kernel_l);
            K = quad_form_diag(K, Sigma);
            for (n in 1:N)
                K[n,n] = K[n,n] + pow(noise_std[n], 2)  + jitter;}
                
        else if (kernel_number==3){
            K = cov_exp(X1, X2, 1., kernel_l, jitter);
            K = quad_form_diag(K, Sigma);
            for (n in 1:N)                            //jitter, add noise only later for K_noise
                K[n,n] = K[n,n] + pow(noise_std[n], 2)  + jitter;}
                
        else if (kernel_number==4){
            K = cov_matern32(X1, X2, 1., kernel_l, jitter);
            K = quad_form_diag(K, Sigma);
            for (n in 1:N)
                K[n,n] = K[n,n] + pow(noise_std[n], 2)  + jitter;}
                
        else if (kernel_number==5){
            K = cov_matern52(X1, X2, 1., kernel_l, jitter);
            K = quad_form_diag(K, Sigma);
            for (n in 1:N)
                K[n,n] = K[n,n] + pow(noise_std[n], 2)  + jitter;}
                
        return K;}
}

data {
    int<lower=1> N;                       // number of stocks
    int<lower=1> D;                       // number of days
    int<lower=1> Q;                       // number of latent dimensions
    matrix[N,D] Y;                        // input Data
    int<lower=1,upper=5> kernel_number;
    real<lower=0> jitter;
}
transformed data {
    vector[N] mu = rep_vector(0, N);
}
parameters {
    vector[Q] X[N];
    vector<lower=0>[N] noise_std;
    real<lower=0> kernel_l;
    matrix<lower=0>[N,D] Sigma;
}
transformed parameters{
    matrix[N,N] L[D];
    
    {matrix[N,N] K[D];
    for (d in 1:D){
        K[d] = kernel_f(X, X, kernel_l, Sigma[:,d], kernel_number, rep_vector(0,N), jitter);
        L[d] = cholesky_decompose(K[d]);
        }
    }
}
model {
    for (n in 1:N)              //latent space process
        X[n] ~ normal(0.0, 1.0);
    
    //for (n in 1:N)
    //    Sigma[n] ~ normal(0.0, 1.0);
    
    // BROWNIAN MOTION VOLATILITY HERE
    
    {
    noise_std ~ normal(0.0, 1.0);
    kernel_l ~ normal(0.0, 1.0);
   } 
    for (d in 1:D)
        col(Y,d) ~ multi_normal_cholesky(mu, L[d]);
    
}
generated quantities{
    matrix[N,D] Y_hat[D];
    matrix[N,N] K_noise[D];
    matrix[N,N] K[D];
    
    for (d in 1:D){
        matrix[N,D] residue;
        K[d] = kernel_f(X, X, kernel_l, Sigma[:,d], kernel_number, rep_vector(0,N), jitter);
        K_noise[d] = kernel_f(X, X, kernel_l, Sigma[:,d], kernel_number, noise_std, jitter);
        Y_hat[d] = K[d] * mdivide_left_spd(K[d], Y);
    }    
}
