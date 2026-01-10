# To-do list

1. Refactor the CMakeLists.txt to include `debug`, `benchmark` and `release` modes separately. Each should point into their own build
files.

`debug`:   
flags: -g, -O0, -fopenmp    
logs: [info], [debug]   
function: debug code during production

`benchmark`:    
flags: -O3, -fopenmp, -ffast-math, -march=native -fopt-info-vec-optimized   
logs: none   
function: allow the testbench to record the time taken

`release`:     
flags: -O3, -fopenmp, -ffast-math, -march=native -fopt-info-vec-optimized 
logs: [info]    
function: allow potential users to use the lib without feeling overwhelmed by the logs