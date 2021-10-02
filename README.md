# A Neural Network in Fortran

Simple feed forward neural network in Fortran

## usage:

```shell
    make
    make run
```

## Create a neural network:

Create a network telling the size (neurons) of earch layer.
```fortran
    nn = new_nn(input_size, output_size, hidden_size)
    # Start training
    call train(nn, inputs, outputs, 10000)
```


