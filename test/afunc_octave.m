## Copyright (C) 2018 james
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} softmax (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: james <james@James-GL752VW>
## Created: 2018-01-25

more off;

1;

function [retVec] = softmax(srcVec)
  # Get size of input vector
  [rows, cols] = size(srcVec);
  
  # Checking
  if(rows > 1)
    error("Input rows > 1");
  endif
  
  expVec = exp(srcVec);
  retVec = expVec / sum(expVec);
endfunction

function [retVec] = softmax_derivative(srcVec)
  # Get size of input vector
  [rows, cols] = size(srcVec);
  
  # Checking
  if(rows > 1)
    error("Input rows > 1");
  endif
  
  # Find softmax derivative
  dstVec = softmax(srcVec);
  retVec = zeros(cols, cols);
  for i = 1 : cols
    for j = 1 : cols
      retVec(i, j) = dstVec(i) * ((i == j) - dstVec(j));
    endfor
  endfor
endfunction

function [retVec] = sigmoid(srcVec)
  # Get size of input vector
  [rows, cols] = size(srcVec);
  
  # Checking
  if(rows > 1)
    error("Input rows > 1");
  endif
  
  # Find sigmoid
  retVec = zeros(1, cols);
  for i = 1 : cols
    retVec(1, i) = 1.0 / (1.0 + exp(-srcVec(1, i)));
  endfor
  
endfunction

function [retVec] = sigmoid_derivative(srcVec)
  # Get size of input vector
  [rows, cols] = size(srcVec);
  
  # Checking
  if(rows > 1)
    error("Input rows > 1");
  endif
  
  # Find sigmoid derivative
  retVec = zeros(cols, cols);
  for i = 1 : cols
    e = exp(-srcVec(1, i));
    retVec(i, i) = e / pow2(1 + e);
  endfor
endfunction

function [retVec] = relu(srcVec)
  # Get size of input vector
  [rows, cols] = size(srcVec);
  
  # Checking
  if(rows > 1)
    error("Input rows > 1");
  endif
  
  # Find sigmoid
  retVec = zeros(1, cols);
  for i = 1 : cols
    retVec(1, i) = max(srcVec(1, i), 0);
  endfor
  
endfunction

function [retVec] = relu_derivative(srcVec)
  # Get size of input vector
  [rows, cols] = size(srcVec);
  
  # Checking
  if(rows > 1)
    error("Input rows > 1");
  endif
  
  # Find sigmoid derivative
  retVec = zeros(cols, cols);
  for i = 1 : cols
    if srcVec(1, i) < 0
      retVec(i, i) = 0;
    else
      retVec(i, i) = 1;
    endif
  endfor
endfunction
