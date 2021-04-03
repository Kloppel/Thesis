functions{
    matrix cov_lin(vector[] X1, vector[] X2, real sigma){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        {
            for (n in 1:N){
                for (m in 1:M){
                    K[n,m] = square(sigma) * X1[n,n] * X2[m,m];
                }
            }
        }
        return K;
    }

    matrix cov_linear(vector[] X1, vector[] X2, real sigma){
        int N = size(X1);
        int M = size(X2);
        int Q = rows(X1[1]);
        matrix[N,M] K;
            {
            matrix[N,Q] x1;
            matrix[M,Q] x2;
            for (n in 1:N)
                x1[n,:] = X1[n]';                //'=Transpose
            for (m in 1:M)
                x2[m,:] = X2[m]';
            K = x1*x2';
        }
        return square(sigma)*K;
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
                for (q in 1:Q)
                    dist = dist + fabs(X1[n,q] - X2[m,q]);
                K[n,m] = square(sigma) * exp(-0.5/l * dist);
            }
        return K;
    }
    
    matrix cov_exp_2(vector[] X1, vector[] X2, real sigma, real l, real jitter){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        int Q = rows(X1[1]);
        for (n in 1:N)
            for (m in 1:M){
                dist = 0;  //sqrt(squared_distance(X1[n], X2[m]) + jitter);
                for (q in 1:Q)
                    dist = dist + square(fabs(X1[n,q] - X2[m,q]));
                K[n,m] = square(sigma) * exp(-0.5/l * dist);
            }
        return K;
    }
    
    matrix cov_matern32(vector[] X1, vector[] X2, real sigma, real l, real jitter){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        int Q = rows(X1[1]);
        for (n in 1:N)
            for (m in 1:M){
                dist = 0;  //sqrt(squared_distance(X1[n], X2[m]) + jitter);
                for (q in 1:Q)
                    dist = dist + fabs(X1[n,q] - X2[m,q]);
                K[n,m] = square(sigma)*(1+sqrt(3)*dist/l)*exp(-sqrt(3)*dist/l);
            }
        return K;
    }
    matrix cov_matern52(vector[] X1, vector[] X2, real sigma, real l, real jitter){
        int N = size(X1);
        int M = size(X2);
        matrix[N,M] K;
        real dist;
        int Q = rows(X1[1]);
        for (n in 1:N)
            for (m in 1:N){
                dist = 0;  //sqrt(squared_distance(X1[n], X2[m]) + jitter);
                for (q in 1:Q)
                    dist = dist + fabs(X1[n,q] - X2[m,q]);
                K[n,m] = square(sigma)*(1+sqrt(5)*dist/l+5*square(dist)/(3*square(l)))*exp(-sqrt(5)*dist/l);
            }
        return K;
    }
    
    matrix kernel_f_time(vector[] X1, vector[] X2, real l, real sigma, real jitter, int kernel_n){
        int D = size(X1);
        matrix[D,D] K;
        
        if (kernel_n==1){
            K = cov_lin(X1, X2, l);
            for (d in 1:D)
                K[d,d] = K[d,d] + pow(sigma, 2) + jitter;
            }

        else if (kernel_n==2){
            K = cov_exp_2(X1, X2, 1., l, jitter);
            for (d in 1:D)
                K[d,d] = K[d,d] + pow(sigma, 2) + jitter;
            }

        else if (kernel_n==3){
            K = cov_exp(X1, X2, 1., l, jitter);
            for (d in 1:D)
                K[d,d] = K[d,d] + pow(sigma, 2) + jitter;
            }

        else if (kernel_n==4){
            K = cov_matern32(X1, X2, 1., l, jitter);
            for (d in 1:D)
                K[d,d] = K[d,d] + pow(sigma, 2) + jitter;
            } 

        else if (kernel_n==5){
            K = cov_matern52(X1, X2, 1., l, jitter);
            for (d in 1:D)
                K[d,d] = K[d,d] + pow(sigma, 2) + jitter;
            }
        return K;
    }
    
    matrix kernel_f_y(vector[] X1, vector[] X2, real l, vector sigma, vector noise, real jitter, int kernel_n){
        int N = size(X1);
        matrix[N,N] K;
        if (kernel_n==1){
                K = cov_linear(X1, X2, l);
                K = quad_form_diag(K, sigma);
                for (n in 1:N){
                    K[n,n] = K[n,n] + pow(noise[n], 2) + jitter;}
                }
                
            else if (kernel_n==2){
                K = cov_exp_2(X1, X2, 1., l, jitter);
                K = quad_form_diag(K, sigma);
                for (n in 1:N){
                    K[n,n] = K[n,n] + pow(noise[n], 2) + jitter;}
                }
                
            else if (kernel_n==3){
                K = cov_exp(X1, X2, 1., l, jitter);
                K = quad_form_diag(K, sigma);
                for (n in 1:N){
                    K[n,n] = K[n,n] + pow(noise[n], 2) + jitter;}
                }
                
            else if (kernel_n==4){
                K = cov_matern32(X1, X2, 1., l, jitter);
                K = quad_form_diag(K, sigma);
                for (n in 1:N){
                    K[n,n] = K[n,n] + pow(noise[n], 2) + jitter;}
                } 
                
            else if (kernel_n==5){
                K = cov_matern52(X1, X2, 1., l, jitter);
                K = quad_form_diag(K, sigma);
                for (n in 1:N){
                    K[n,n] = K[n,n] + pow(noise[n], 2) + jitter;}
                }
            return K;
    } 
}

data {
    // data and model parameters
    int<lower=1> N;
    int<lower=1> D;
    int<lower=1> Q;
    matrix[N,D] Y;
    int<lower=1,upper=5> kernel_number_x;
    int<lower=1,upper=5> kernel_number_y;
    real<lower=0> jitter;
    
    // lengthscale dist parameters
    real<lower=0> alpha_x;
    real<lower=0> beta_x;
    real<lower=0> alpha_y;
    real<lower=0> beta_y;
    
    // standard deviation parameters
    real <lower=0,upper=1> std_x;
    real <lower=0,upper=0.1> noise_x;
    real <lower=0,upper=1> std_y;
    real <lower=0,upper=0.1> noise_y;
}

transformed data {
    vector[N] zeros_N = rep_vector(0, N);
    vector[D] zeros_D = rep_vector(0, D);
    vector[D] time[D];
    for (d in 1:D){
        time[d] = rep_vector(d,d);
        }
}

parameters {
    // X
    vector[Q] X[D,N];
    real<lower=0> kernel_lengthscale_x;
    real<lower=0> kernel_std_x;
    real<lower=0> noise_std_x;
    real<lower=2> t_nu_x;
    
    // Y
    real<lower=0> kernel_lengthscale_y;
    vector<lower=0>[N] diag_std_y;
    vector<lower=0>[N] noise_std_y;
    real<lower=2> t_nu_y;
    
}

transformed parameters {
    real R2 = mean(square(diag_std_y)./(square(diag_std_y)+square(noise_std_y)));
}

model {
    // X
    kernel_lengthscale_x ~ gamma(alpha_x, beta_x);
    kernel_std_x ~ normal(0, std_x);
    noise_std_x ~ normal(0, noise_x);

    // Y
    kernel_lengthscale_y ~ inv_gamma(alpha_y, beta_y);
    diag_std_y ~ normal(0, std_y);
    noise_std_y ~ normal(0, noise_y);


    //prior on X
    {
        matrix[D,D] K_x;
        K_x = kernel_f_time(time, time, kernel_lengthscale_x, noise_std_x, jitter, kernel_number_x);

        for (q in 1:Q)
            for (n in 1:N)
                to_vector(X[:,n,q]) ~ multi_student_t(t_nu_x, zeros_D, K_x);
    }


    //likelihood
    {
        matrix[N,N] K_y[D];
        //matrix[N,N] K_y_;
        for (d in 1:D) {
            K_y[d] = kernel_f_y(X[d], X[d], kernel_lengthscale_y, diag_std_y, rep_vector(0,N), jitter, kernel_number_y);
        }
        for (d in 1:D) 
            col(Y,d) ~ multi_student_t(t_nu_y, zeros_N, K_y[d]);
    }
}

generated quantities {
    // real log_likelihood = 0;   
    // real R2_hat_N = 0;
    // vector[N] R2_hat_vec_N;
    // matrix[N,D] Y_hat[D];
    matrix[N,N] K_y[D];
    // matrix[N,N] K_y[D];
    // matrix[N,N] K_y_noise[D];
    
    for (d in 1:D){
        K_y[d] = kernel_f_y(X[d], X[d], kernel_lengthscale_y, diag_std_y, rep_vector(0,N), jitter, kernel_number_y);
        //K_y_noise[d] = kernel_f_y(X[d], X[d], kernel_lengthscale_y, diag_std_y, noise_std_y, jitter, kernel_number_y);
        //Y_hat[d] = K_y[d] * mdivide_left(K_y_noise[d], Y);
        }

    //for (d in 1:D){
    //    log_likelihood = log_likelihood + multi_student_t_lpdf(col(Y,d) | zeros_N, L_y[d]);}
}
