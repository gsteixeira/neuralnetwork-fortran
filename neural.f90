!
!  A simple feed forward XOR neural network in Fortran
!
!       Author: Gustavo Selbach Teixeira
!

! The layer object definition
module layer_mod
    implicit none
    type :: Layer
        real(kind=4), dimension(:), allocatable :: values
        real(kind=4), dimension(:), allocatable :: bias
        real(kind=4), dimension(:), allocatable :: deltas
!             real(kind=4), dimension(4:4) :: weights
        real(kind=4), dimension(:,:), allocatable :: weights
        integer :: n_nodes
        integer :: n_synapses
    end type Layer
    interface Layer
        procedure :: new_layer
    end interface Layer
    contains
        ! The Layer constructor method
        type(Layer) function new_layer (n_nodes, n_synapses)
            integer, intent(in) :: n_nodes
            integer, intent(in) :: n_synapses
            integer :: i, k
            new_layer%n_nodes = n_nodes
            new_layer%n_synapses = n_synapses
            if (.not. allocated(new_layer%values)) then
                allocate(new_layer%values(n_nodes))
                allocate(new_layer%bias(n_nodes))
                allocate(new_layer%deltas(n_nodes))
                allocate(new_layer%weights(n_synapses, n_nodes))
            end if
            ! initialize with random numbers
            do i = 1, n_nodes
                call random_number(new_layer%values(i))
                call random_number(new_layer%bias(i))
                call random_number(new_layer%deltas(i))
            end do
            do i = 1, n_synapses
                do k = 1, n_nodes
                    call random_number(new_layer%weights(k, i))
                end do
            end do
        end function new_layer
end module layer_mod

! The neural network object definition
module neural_network_mod
    use layer_mod
    implicit none
    type :: NeuralNetwork
        type(Layer) :: input_layer
        type(Layer) :: hidden_layer
        type(Layer) :: output_layer
        real(kind=4) :: learning_rate
        integer :: input_size
        integer :: hidden_size
        integer :: output_size
    end type NeuralNetwork
    interface NeuralNetwork
        procedure :: new_nn
    end interface NeuralNetwork
    contains
        ! The NeuralNetwork constructor method
        type(NeuralNetwork) function new_nn(input_size, output_size, hidden_size)
            integer, intent(in) :: input_size
            integer, intent(in) :: output_size
            integer, intent(in) :: hidden_size
            new_nn%input_layer = Layer(input_size, 1)
            new_nn%hidden_layer = Layer(hidden_size, input_size)
            new_nn%output_layer = Layer(output_size, hidden_size)
            
            new_nn%input_size = input_size
            new_nn%hidden_size = hidden_size
            new_nn%output_size = output_size
            new_nn%learning_rate = 0.1
        end function
end module neural_network_mod

! Train the network, main loop
subroutine train(nn, inputs, outputs, iteractions)
    use neural_network_mod
    use layer_mod
    type(NeuralNetwork), intent(inout) :: nn
    real(kind=4), dimension(4, 2), intent(in) :: inputs
    real(kind=4), dimension(4, 1), intent(in) :: outputs
    integer, intent(in) :: iteractions
    integer :: i, e
    do e = 1, iteractions
        do i = 1, 4
            call set_inputs(nn, inputs(i, :))
            call forward_pass(nn)
            print *, "Input ", inputs(i,:), " Expected ", outputs(i,:), " Output: ", nn%output_layer%values(1)
            ! compute deltas
            call calc_delta_output(nn, outputs(i,:))
            call calc_deltas(nn%output_layer, nn%hidden_layer)
            ! update weights and bias
            call update_weights(nn%output_layer, nn%hidden_layer, nn)
            call update_weights(nn%hidden_layer, nn%input_layer, nn)
        end do
    end do
end subroutine

! The forward pass prediction process
subroutine forward_pass(nn)
    use neural_network_mod
    type(NeuralNetwork), intent(inout) :: nn
    call activation_function(nn, nn%input_layer, nn%hidden_layer)
    call activation_function(nn, nn%hidden_layer, nn%output_layer)
end subroutine

! Feed data to the network
subroutine set_inputs(nn, inputs)
    use neural_network_mod
    type(NeuralNetwork), intent(inout) :: nn
    real(kind=4), dimension(2), intent(in) :: inputs
    integer :: i
    !
    do i = 1, nn%input_size
        nn%input_layer%values(i) = inputs(i)
    end do
end subroutine

! Sigmoid function
real(kind=4) function sigmoid(x)
    implicit none
    real(kind=4), intent(in) :: x
    sigmoid = 1 / (1 + exp(-x))
end function

! Diferential simoid function
real(kind=4) function d_sigmoid(x)
    implicit none
    real(kind=4), intent(in) :: x
    d_sigmoid = x * (1 - x)
end function

! The Activation function
subroutine activation_function(nn, source, ltarget)
    use neural_network_mod
    use layer_mod
    type(NeuralNetwork), intent(inout) :: nn
    type(Layer), intent(inout) :: source
    type(Layer), intent(inout) :: ltarget
    real(kind=4) :: sigmoid
    integer :: i, k
    real(kind=4) :: activation
    !
    do i = 1, ltarget%n_nodes
        activation = ltarget%bias(i)
        do k = 1, source%n_nodes
            activation = activation + (source%values(k) * ltarget%weights(k, i))
        end do
        ltarget%values(i) = sigmoid(activation)
    end do
end subroutine

! Compute the delta for the output layer
subroutine calc_delta_output(nn, expected)
    use neural_network_mod
    type(NeuralNetwork), intent(inout) :: nn
    real(kind=4), dimension(1), intent(in) :: expected
    integer :: i
    real(kind=4) :: error
    real(kind=4) :: d_sigmoid
    do i = 1, nn%output_layer%n_nodes
        error = (expected(i) - nn%output_layer%values(i))
        nn%output_layer%deltas(i) = (error * d_sigmoid(nn%output_layer%values(i)))
    end do
end subroutine

! compute the deltas though the layers
subroutine calc_deltas(source, ltarget)
    use layer_mod
    type(Layer), intent(inout) :: source
    type(Layer), intent(inout) :: ltarget
    integer :: i, j
    real(kind=4) :: error
    real(kind=4) :: d_sigmoid
    do i = 1, ltarget%n_nodes
        error = 0.0
        do k = 1, source%n_nodes
            error = error + (source%deltas(k) * source%weights(i, k))
        end do
        ltarget%deltas(i) = (error * d_sigmoid(ltarget%values(i)))
    end do
end subroutine

! Update the weights and bias
subroutine update_weights(source, ltarget, nn)
    use layer_mod
    use neural_network_mod
    type(Layer), intent(inout) :: source
    type(Layer), intent(inout) :: ltarget
    type(NeuralNetwork), intent(in) :: nn
    integer :: i, k
    do i = 1, source%n_nodes
        source%bias(i) = source%bias(i) + (source%deltas(i) * nn%learning_rate)
        do k = 1, ltarget%n_nodes
            source%weights(k, i) = source%weights(k, i) + (ltarget%values(k) * source%deltas(i) * nn%learning_rate)
        end do
    end do
end subroutine

! Main program
program nnetwork
    use layer_mod
    use neural_network_mod
    implicit none
    
    type(NeuralNetwork) :: nn
    real(kind=4), dimension(4, 2) :: inputs
    real(kind=4), dimension(4, 1) :: outputs
    ! seed the random generator
    call random_init(.true., .true.)
    ! Set the training parameters
    inputs(1, :) = (/0, 0/)
    inputs(2, :) = (/1, 0/)
    inputs(3, :) = (/0, 1/)
    inputs(4, :) = (/1, 1/)
    
    outputs(1, :) = (/0/)
    outputs(2, :) = (/1/)
    outputs(3, :) = (/1/)
    outputs(4, :) = (/0/)
    ! Create the network
    nn = new_nn(2, 1, 4)
    call train(nn, inputs, outputs, 10000)
end program nnetwork
