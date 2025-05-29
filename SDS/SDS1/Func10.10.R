# Linear Congruential Generator with the ANSI C default (Saucier, 2000)
lcg <- function(n = 100, M = 2^32, a = 12345, b = 1103515245, u0 = as.numeric(Sys.time()) * 100) {
  out <- rep(NA, n)  # Allocate output
  # Main loop
  for (i in 1:n) {
    u0 <- (a + b * u0) %% M
    out[i] <- u0 / (M + 1)
  }
  return(out)
}

myqexp <- function(u, alpha = 1) -log(1-u)/alpha
