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

# Iris dataset
dataset = [
  [5.1  3.5  1.4  0.2  1.0  0.0  0.0]
  [4.9  3.0  1.4  0.2  1.0  0.0  0.0]
  [4.6  3.1  1.5  0.2  1.0  0.0  0.0]
  [5.0  3.6  1.4  0.2  1.0  0.0  0.0]
  [5.4  3.9  1.7  0.4  1.0  0.0  0.0]
  [5.0  3.4  1.5  0.2  1.0  0.0  0.0]
  [4.4  2.9  1.4  0.2  1.0  0.0  0.0]
  [4.8  3.4  1.6  0.2  1.0  0.0  0.0]
  [4.8  3.0  1.4  0.1  1.0  0.0  0.0]
  [4.3  3.0  1.1  0.1  1.0  0.0  0.0]
  [5.7  4.4  1.5  0.4  1.0  0.0  0.0]
  [5.4  3.9  1.3  0.4  1.0  0.0  0.0]
  [5.1  3.5  1.4  0.3  1.0  0.0  0.0]
  [5.4  3.4  1.7  0.2  1.0  0.0  0.0]
  [5.1  3.7  1.5  0.4  1.0  0.0  0.0]
  [4.6  3.6  1.0  0.2  1.0  0.0  0.0]
  [5.1  3.3  1.7  0.5  1.0  0.0  0.0]
  [4.8  3.4  1.9  0.2  1.0  0.0  0.0]
  [5.0  3.0  1.6  0.2  1.0  0.0  0.0]
  [5.0  3.4  1.6  0.4  1.0  0.0  0.0]
  [5.2  3.5  1.5  0.2  1.0  0.0  0.0]
  [5.2  3.4  1.4  0.2  1.0  0.0  0.0]
  [4.7  3.2  1.6  0.2  1.0  0.0  0.0]
  [4.8  3.1  1.6  0.2  1.0  0.0  0.0]
  [5.4  3.4  1.5  0.4  1.0  0.0  0.0]
  [5.2  4.1  1.5  0.1  1.0  0.0  0.0]
  [5.5  4.2  1.4  0.2  1.0  0.0  0.0]
  [5.0  3.2  1.2  0.2  1.0  0.0  0.0]
  [5.5  3.5  1.3  0.2  1.0  0.0  0.0]
  [5.1  3.4  1.5  0.2  1.0  0.0  0.0]
  [5.0  3.5  1.3  0.3  1.0  0.0  0.0]
  [4.5  2.3  1.3  0.3  1.0  0.0  0.0]
  [5.1  3.8  1.9  0.4  1.0  0.0  0.0]
  [4.8  3.0  1.4  0.3  1.0  0.0  0.0]
  [4.6  3.2  1.4  0.2  1.0  0.0  0.0]
  [5.3  3.7  1.5  0.2  1.0  0.0  0.0]
  [5.0  3.3  1.4  0.2  1.0  0.0  0.0]
  [7.0  3.2  4.7  1.4  0.0  1.0  0.0]
  [6.4  3.2  4.5  1.5  0.0  1.0  0.0]
  [6.9  3.1  4.9  1.5  0.0  1.0  0.0]
  [5.5  2.3  4.0  1.3  0.0  1.0  0.0]
  [6.3  3.3  4.7  1.6  0.0  1.0  0.0]
  [4.9  2.4  3.3  1.0  0.0  1.0  0.0]
  [6.0  2.2  4.0  1.0  0.0  1.0  0.0]
  [6.1  2.9  4.7  1.4  0.0  1.0  0.0]
  [5.6  2.9  3.6  1.3  0.0  1.0  0.0]
  [6.7  3.1  4.4  1.4  0.0  1.0  0.0]
  [5.6  3.0  4.5  1.5  0.0  1.0  0.0]
  [5.8  2.7  4.1  1.0  0.0  1.0  0.0]
  [6.2  2.2  4.5  1.5  0.0  1.0  0.0]
  [5.6  2.5  3.9  1.1  0.0  1.0  0.0]
  [5.9  3.2  4.8  1.8  0.0  1.0  0.0]
  [6.1  2.8  4.0  1.3  0.0  1.0  0.0]
  [6.3  2.5  4.9  1.5  0.0  1.0  0.0]
  [6.1  2.8  4.7  1.2  0.0  1.0  0.0]
  [6.4  2.9  4.3  1.3  0.0  1.0  0.0]
  [6.6  3.0  4.4  1.4  0.0  1.0  0.0]
  [6.7  3.0  5.0  1.7  0.0  1.0  0.0]
  [5.7  2.6  3.5  1.0  0.0  1.0  0.0]
  [5.5  2.4  3.7  1.0  0.0  1.0  0.0]
  [5.8  2.7  3.9  1.2  0.0  1.0  0.0]
  [6.0  2.7  5.1  1.6  0.0  1.0  0.0]
  [5.4  3.0  4.5  1.5  0.0  1.0  0.0]
  [6.7  3.1  4.7  1.5  0.0  1.0  0.0]
  [6.3  2.3  4.4  1.3  0.0  1.0  0.0]
  [5.5  2.6  4.4  1.2  0.0  1.0  0.0]
  [6.1  3.0  4.6  1.4  0.0  1.0  0.0]
  [5.8  2.6  4.0  1.2  0.0  1.0  0.0]
  [5.0  2.3  3.3  1.0  0.0  1.0  0.0]
  [5.6  2.7  4.2  1.3  0.0  1.0  0.0]
  [5.7  3.0  4.2  1.2  0.0  1.0  0.0]
  [5.7  2.9  4.2  1.3  0.0  1.0  0.0]
  [6.2  2.9  4.3  1.3  0.0  1.0  0.0]
  [6.3  3.3  6.0  2.5  0.0  0.0  1.0]
  [5.8  2.7  5.1  1.9  0.0  0.0  1.0]
  [7.1  3.0  5.9  2.1  0.0  0.0  1.0]
  [6.5  3.0  5.8  2.2  0.0  0.0  1.0]
  [7.6  3.0  6.6  2.1  0.0  0.0  1.0]
  [4.9  2.5  4.5  1.7  0.0  0.0  1.0]
  [7.3  2.9  6.3  1.8  0.0  0.0  1.0]
  [6.7  2.5  5.8  1.8  0.0  0.0  1.0]
  [6.5  3.2  5.1  2.0  0.0  0.0  1.0]
  [6.4  2.7  5.3  1.9  0.0  0.0  1.0]
  [6.8  3.0  5.5  2.1  0.0  0.0  1.0]
  [5.8  2.8  5.1  2.4  0.0  0.0  1.0]
  [6.4  3.2  5.3  2.3  0.0  0.0  1.0]
  [7.7  3.8  6.7  2.2  0.0  0.0  1.0]
  [5.6  2.8  4.9  2.0  0.0  0.0  1.0]
  [7.7  2.8  6.7  2.0  0.0  0.0  1.0]
  [6.3  2.7  4.9  1.8  0.0  0.0  1.0]
  [6.7  3.3  5.7  2.1  0.0  0.0  1.0]
  [7.2  3.2  6.0  1.8  0.0  0.0  1.0]
  [6.1  3.0  4.9  1.8  0.0  0.0  1.0]
  [6.4  2.8  5.6  2.1  0.0  0.0  1.0]
  [6.4  2.8  5.6  2.2  0.0  0.0  1.0]
  [6.3  2.8  5.1  1.5  0.0  0.0  1.0]
  [6.1  2.6  5.6  1.4  0.0  0.0  1.0]
  [7.7  3.0  6.1  2.3  0.0  0.0  1.0]
  [6.3  3.4  5.6  2.4  0.0  0.0  1.0]
  [6.4  3.1  5.5  1.8  0.0  0.0  1.0]
  [6.0  3.0  4.8  1.8  0.0  0.0  1.0]
  [6.9  3.1  5.4  2.1  0.0  0.0  1.0]
  [6.7  3.1  5.6  2.4  0.0  0.0  1.0]
  [6.9  3.1  5.1  2.3  0.0  0.0  1.0]
  [6.8  3.2  5.9  2.3  0.0  0.0  1.0]
  [6.7  3.3  5.7  2.5  0.0  0.0  1.0]
  [6.5  3.0  5.2  2.0  0.0  0.0  1.0]
  [6.2  3.4  5.4  2.3  0.0  0.0  1.0]
  [5.9  3.0  5.1  1.8  0.0  0.0  1.0]
  [4.7  3.2  1.3  0.2  1.0  0.0  0.0]
  [5.8  4.0  1.2  0.2  1.0  0.0  0.0]
  [4.4  3.2  1.3  0.2  1.0  0.0  0.0]
  [6.6  2.9  4.6  1.3  0.0  1.0  0.0]
  [6.8  2.8  4.8  1.4  0.0  1.0  0.0]
  [6.0  2.9  4.5  1.5  0.0  1.0  0.0]
  [6.0  3.4  4.5  1.6  0.0  1.0  0.0]
  [6.5  3.0  5.5  1.8  0.0  0.0  1.0]
  [6.0  2.2  5.0  1.5  0.0  0.0  1.0]
  [6.9  3.2  5.7  2.3  0.0  0.0  1.0]
  [6.2  2.8  4.8  1.8  0.0  0.0  1.0]
];

# === MAIN ===
INPUTS = 4
OUTPUTS = 3
ITER = 1000
L_RATE = 0.05

# Rand weight and zero bias
weight = rand(INPUTS, OUTPUTS);
weight = weight * 2 - 1
bias = zeros(1, OUTPUTS)

# Check size
[dataRows, dataCols] = size(dataset);
if dataCols != INPUTS + OUTPUTS
  error("Dataset column size does not match with the model setting");
endif

# Softmax regression training
for iter = 1 : ITER
  mse = 0;
  hit = 0;
  for i = 1 : dataRows
    # Forward
    input = dataset(i, 1 : INPUTS);
    desire = dataset(i, INPUTS + 1 : dataCols);
    
    tmpOut = input * weight + bias;
    output = softmax(tmpOut);
    
    [d, dIndex] = max(desire);
    [o, oIndex] = max(output);
    if dIndex == oIndex
      hit++;
    endif
    
    # Find error
    gradVec = desire - output;
    mse += sum(gradVec * transpose(gradVec)) / OUTPUTS;
    
    # Backpropagation
    sGrad = softmax_derivative(tmpOut);
    bGrad = gradVec * sGrad;
    bias += L_RATE * bGrad;
    
    wGrad = transpose(input) * bGrad;
    weight += L_RATE * wGrad;
    
  endfor

  mse /= OUTPUTS * dataRows;
  hit /= dataRows;
  printf("iter %d, mse: %f, accuracy: %f %%\n", iter, mse, hit * 100)
endfor