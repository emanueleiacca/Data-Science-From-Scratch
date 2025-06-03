str(loyn)
head(loyn)


model1_string = "
  model {
  #Likelihood
  for (i in 1:n) {
  ABUND[i]~dnorm(mu[i],tau)
  mu[i] <- beta[1] + beta[2]*AREA[i] + beta[3]*YR.ISOL[i] + beta[4]*DIST[i] + beta[5]*LDIST[i] + beta[6]*GRAZE[i] + beta[7]*ALT[i]
  }
  #Priors
  for (j in 1:n_coeff) {
  beta[j] ~ dnorm(0.01,1.0E-6)
  }
  tau <- 1/pow(sigma,2)
  sigma~dunif(0,100)
  }
  "
params <- c("beta", "sigma")

loyn.list1=list(n=nrow(loyn),
                n_coeff=ncol(loyn),
                ABUND=loyn$ABUND,
                AREA=loyn$AREA,
                YR.ISOL=loyn$YR.ISOL,
                DIST=loyn$DIST,
                LDIST=loyn$LDIST,
                GRAZE=loyn$GRAZE,
                ALT=loyn$ALT
)

library(R2jags)
loyn_mod1 <- jags(data = loyn.list1,
                  inits = NULL, parameters.to.save = params,
                  model.file = textConnection(model1_string), n.chains = 3, n.iter = 33000,
                  n.burnin = 3000, n.thin = 10)
