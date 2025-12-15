using Test

function runtests()
    #include("hello_world_test.jl")
    include("burgers_weno_test.jl")
end

runtests()