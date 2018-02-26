function A_inv_b = matrixInverseVector(A, b, x_init, alpha)
  % Your code here
  diff = 100
  while diff > 0.01,
    p = 2 * A
    q = A * x_init - b

    xnew = x_init - alpha * p * q
    diff = squaredErrorCost(A, b, xnew)
    x_init = xnew
  end;
  A_inv_b = xnew
  return;
endfunction
