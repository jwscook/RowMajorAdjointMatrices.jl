using RowMajorAdjointMatrices

using Equilibrium
using ForwardDiff
using LinearAlgebra
using Test

@testset "RowMajorAdjointMatrix" begin
  @testset "solves" begin
    @testset "rectanglar" begin
      N, M = 11, 10
      matrix = zeros(ComplexF64, M, N) # note order M then N
      rmam = RowMajorAdjointMatrix(matrix)
      
      actualmatrix = zeros(ComplexF64, N, M) # note order N then M
      for i in 1:N, j in 1:M # note that the loops are in the wrong order!
        ijv = rand(ComplexF64)
        rmam[i, j] = ijv
        actualmatrix[i, j] = ijv
      end
      @test rmam.matrix' ≈ actualmatrix
      
      b = rand(ComplexF64, N)
      
      x = actualmatrix \ b
      y = rmam \ b
      @test x ≈ y
     
      yview = rmam[:, :] \ b
      @test x ≈ yview

      rmamcopy = deepcopy(rmam)
      matrixcopyconjtransp = deepcopy(rmamcopy.matrix)'

      @testset "lq" begin
        rmamcopy.matrix .= rmam.matrix
        matrixcopyconjtransp .= rmamcopy.matrix'
        @test Matrix(lq(matrixcopyconjtransp)) ≈ Matrix(lq(rmamcopy.matrix'))
        @test x ≈ rmamcopy.matrix' \ b
        rmam1 = deepcopy(rmamcopy)
        @test x ≈ lq!(rmamcopy) \ b
        rmamfact = lq!(rmam1)
        #@test x ≈ rmamfact.factorisation.Q * (b' / rmamfact.factorisation.R)'
        @test !(x ≈ rmam1 \ b) # rmam1 has been changed in-place
        # rmamcopy has been changed in-place
        @test !(matrixcopyconjtransp ≈ rmamcopy.matrix')
      end

    end
  
    @testset "square" begin
      N, M = 10, 10
      matrix = zeros(ComplexF64, M, N) # note order M then N
      rmam = RowMajorAdjointMatrix(matrix)
      
      actualmatrix = zeros(ComplexF64, N, M) # note order N then M
      for i in 1:N, j in 1:M # note that the loops are in the wrong order!
        ijv = rand(ComplexF64)
        rmam[i, j] = ijv
        actualmatrix[i, j] = ijv
      end
      @test rmam.matrix' ≈ actualmatrix
      
      b = rand(ComplexF64, N)

      x = actualmatrix \ b
      y = rmam \ b
      @test x ≈ y

      yview = rmam[:, :] \ b
      @test x ≈ yview

      rmamcopy = deepcopy(rmam)
      #@test x ≈ IterativeSolvers.gmres(actualmatrix, b)
      #@test x ≈ IterativeSolvers.gmres(rmamcopy.matrix', b)
      #@test x ≈ IterativeSolvers.gmres(rmamcopy, b)
      matrixcopyconjtransp = deepcopy(rmamcopy.matrix)'
      @test Matrix(lu(matrixcopyconjtransp)) ≈ Matrix(lu(rmamcopy.matrix'))
      @test x ≈ lu!(matrixcopyconjtransp) \ b
      @test !(matrixcopyconjtransp ≈ rmamcopy.matrix')
      @test x ≈ lu(rmamcopy.matrix') \ b
      @test x ≈ rmamcopy.matrix' \ b
      rmam1 = deepcopy(rmamcopy)
      @test x ≈ lu!(rmamcopy) \ b
      rmamfact = lu!(rmam1)
      #@test x ≈ ((b' / rmamfact.factorisation.U) / rmamfact.factorisation.L)'
      @test !(x ≈ rmam1 \ b) # rmam1 has been changed in-place
      # rmamcopy has been changed in-place
      @test !(matrixcopyconjtransp ≈ rmamcopy.matrix')
    end
  end
  @testset "populate by 3x3" begin
    matrix = zeros(ComplexF64, 3, 3)
    rmam = RowMajorAdjointMatrix(matrix)
    try
      m = rand(ComplexF64, 3, 3)
      rmam[1:3, 1:3] .= m # insertion tranpose-conjugates
      @test all(rmam[1:3, 1:3] .== m)
      @test all(rmam.matrix .≈ m')
      @test true
    catch err
      @show err
      @test false
    end
  end
  @testset "view" begin
    matrix = rand(ComplexF64, 2, 2)
    @assert all(!, matrix .≈ matrix')
    rmam = RowMajorAdjointMatrix(matrix)
    @test all(rmam .≈ rmam[:, :])
    @test all((@view rmam[:, :]) .≈ rmam[:, :])
    @test all((@view rmam[:, :])' .≈ matrix)
  end
  @testset "Colon" begin
    matrix = rand(ComplexF64, 3, 3)
    rmam = RowMajorAdjointMatrix(conj.(transpose(deepcopy(matrix))))
    for ij in 1:3
      @test all(conj.(rmam[ij, :]') .== matrix[ij, :])
      @test all(conj.(rmam[:, ij]') .== matrix[:, ij])
    end
  end
end

