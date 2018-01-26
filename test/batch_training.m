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
HIDDEN = 12
OUTPUTS = 3
ITER = 10000
L_RATE = 0.01

# Rand weight and zero bias
hWeight = rand(INPUTS, HIDDEN);
hWeight = hWeight * 2 - 1
hBias = zeros(1, HIDDEN)

oWeight = rand(HIDDEN, OUTPUTS);
oWeight = oWeight * 2 - 1
oBias = zeros(1, OUTPUTS)

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
    
    tmpHOut = input * hWeight + hBias;
    hOut = sigmoid(tmpHOut);
    
    tmpOOut = hOut * oWeight + oBias;
    output = sigmoid(tmpOOut);
    
    [d, dIndex] = max(desire);
    [o, oIndex] = max(output);
    if dIndex == oIndex
      hit++;
    endif
    
    # Find error
    gradVec = desire - output;
    mse += sum(gradVec * transpose(gradVec)) / OUTPUTS;
    
    # Backpropagation
    grad0 = desire - output;
    grad1 = grad0 * sigmoid_derivative(tmpOOut);
    
    oBias += L_RATE * grad1;
    oWeight += L_RATE * (transpose(hOut) * grad1);
    
    grad2 = grad1 * transpose(oWeight);
    grad3 = grad2 * sigmoid_derivative(tmpHOut);
    
    hBias += L_RATE * grad2;
    hWeight += L_RATE * (transpose(input) * grad3);
    
  endfor

  mse /= dataRows;
  hit /= dataRows;
  printf("iter %d, mse: %f, accuracy: %f %%\n", iter, mse, hit * 100)
endfor