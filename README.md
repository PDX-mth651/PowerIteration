# PowerIteration
An example use of the linear algebra library.

# Build
* Make the linear algebra library found at https://github.com/gelever/linalgcpp.
* Copy ```Makefile.in``` to ```Makefile```.
* Modify ```Makefile``` so that the linalgcpp directory path points to where it is located on your machine.
* Call ```make``` and try the example program in build/poweriter
# Output
Program ouput should read:
```
Input:
     2    -1    -1     0     0
    -1     2    -1     0     0
    -1    -1     4    -1    -1
     0     0    -1     2    -1
     0     0    -1    -1     2

Max Eval Dense:  5
Max Eval Sparse: 5
Max Eval Coo:    5
```
