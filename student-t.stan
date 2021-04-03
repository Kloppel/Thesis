functions {
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
        return square(sigma)*K;
    }
    
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
        return K;
    }
    
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
        return K;
    }
    
    matrix cov_exp_l2(vector[] X1, vector[] X2, real sigma, real l, real jitter){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        for (n in 1:N)
            for (m in 1:M){
                dist = sqrt(squared_distance(X1[n], X2[m]) + jitter);
                K[n,m] = square(sigma) * exp(-0.5/l * dist);
            }
        return K;
    }
    
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
        return K;
    }
    
    matrix kernel_f(vector[] X1, vector[] X2, real sigma, real l, 
                    real a, int kernel, vector diag_stds, real jitter){
        // X:latent space, sigma:kernel_std, l:lengthscale, f:frequency, a:alpha
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        if (kernel==1)
            K = cov_linear(X1, X2, a);               // K = a^2*X1*X2.T
        else if (kernel==2){
            K = cov_exp_quad(X1, X2, sigma, l);
            for (n in 1:N)
                K[n,n] = K[n,n] + jitter;
            K = quad_form_diag(K, diag_stds);
        }
        else if (kernel==3){
            K = cov_exp(X1, X2, sigma, l, jitter);
            K = quad_form_diag(K, diag_stds);
        }
        else if (kernel==4){
            K = cov_matern32(X1, X2, sigma, l, jitter);
            K = quad_form_diag(K, diag_stds);
        }
        else if (kernel==5){
            K = cov_matern52(X1, X2, sigma, l, jitter);
            K = quad_form_diag(K, diag_stds);
        }
        return K;
    }
}

data {
    int<lower=1> N;                       // number of stocks
    int<lower=1> D;                       // number of days
    int<lower=1> Q;                       // number of latent dimensions
    matrix[N,D] Y;                        // input Data
    int<lower=1,upper=5> kernel;    // used by function 'kernel_f()' for model choice
    real<lower=0> jitter;
}

transformed data {
    vector[N] mu = rep_vector(0, N);      // mean vector, assumed to be 0 in good estimation
}

parameters {
    vector[Q] X[N];                       // latent space
    real<lower=0> kernel_lengthscale;     // kernel lengthscale
    vector<lower=0>[N] diag_stds;         // standard deviation for each stock
    vector<lower=0>[N] noise_std;         // observation noise ... non isotropic a la factor model
    real<lower=0> alpha;                  // kernel std for linear kernel
    real<lower=2> t_nu;                          // degrees of freedom of student t distribution
}

transformed parameters {
    //matrix[N,N] K;
    real R2 = 0;
    // we set kernel_std to 1. and model different stds for different points by diag_stds
    matrix[N,N] K = kernel_f(X, X, 1., kernel_lengthscale, alpha, kernel, diag_stds, jitter);
                                         // fills covariance matrices with corresponding kernel function arguments
    
    for (n in 1:N)
        K[n,n] = K[n,n] + pow(noise_std[n], 2) + jitter; 
                                        // make student-t covariance matrix here. either define K outside of this function, and 
                                        // use transforming steps outside, which decreases performance, or accept correlation between  
                                        // scaling matrices.
        
    R2 = sum(1 - square(noise_std) ./diagonal(K) )/N;
}

model {
    diag_stds ~ normal(0, .5);
    noise_std ~ normal(0, .5);                 // noise in kernel
    kernel_lengthscale ~ inv_gamma(3.0,1.0);// inv_gamma for zero-avoiding prop
    alpha ~ inv_gamma(3.0,1.0);                  // kernel std for linear kernel
    
    for (n in 1:N)                               //latent space process
        X[n] ~ normal(0, 1);
        
    for (d in 1:D)                               // likelihood choice
        col(Y,d) ~ multi_student_t(t_nu, mu, K); 
}

generated quantities {
    real log_likelihood = 0;   //just the log_likelihood values. without log_prior
    //real R2_hat_N = 0;
    //vector[N] R2_hat_vec_N;
    //matrix[N,D] Y_hat;         
    
    //matrix[N,N] K = kernel_f(X, X, 1., kernel_lengthscale, alpha, kernel, diag_stds, jitter);
    
    for (d in 1:D)
        log_likelihood = log_likelihood + multi_student_t_lpdf(col(Y,d) |t_nu, mu, K); 
        
    //{
    //    matrix[N,N] K_noise = K_out;
    //    matrix[N,D] resid;
    //    
    //    for (n in 1:N)
    //        K_noise[n,n] = K_noise[n,n] + pow(noise_std[n], 2) + jitter;
    //    Y_hat = K_out * mdivide_left_spd(K_noise, Y);
    //    resid = Y - Y_hat;
    //    for (n in 1:N)
    //        R2_hat_vec_N[n] = 1 - sum( square(row(resid,n)) )/ sum( square(row(Y,n)-mean(row(Y,n))) );
    //    
    //    R2_hat_N = mean(R2_hat_vec_N);
    //    K_out = K_noise;           //includes non-isotropic noise to the output K 
    //}
}
