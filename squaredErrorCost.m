function cost = squaredErrorCost(A, b, x)
  % Your code here
  c = A*x - b;
  cost = sum(c .^ 2);
endfunction
