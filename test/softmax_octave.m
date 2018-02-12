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

# === MAIN ===
srcVec = [1, 2, 3, 4, 5]
grad = softmax_derivative(srcVec)
gradT = transpose(grad)
if grad == gradT
  printf("grad == gradT\n")
endif