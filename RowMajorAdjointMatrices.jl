module RowMajorAdjointMatrix

using LinearAlgebra

"""
    RowMajorAdjointMatrix(::Type{T},numrows,numcols)whereT

The RowMajorAdjointMatrix exists because the lhs matrix
is better populated in a row major fashion. This is because
the unknowns are in Fourier space, but it is computationally
more effective to calculate the spatial parameters once per
(R,Z) location and then use them for all Fourier components.
This leads to inefficient access of the matrix but this isn't
too bad if the matrix is in RAM because the inner loop is so
expensive. However, once the matrix is memory mapped to the SSD
then it's inordinately expensive to hop around the matrix.

Instead of solving A x = b via x = A \\ b we must solve
x = (b' / Ã)' where Ã is the RowMajorAdjointMatrix, which naturally
is the conjugate tranpose of A.

Note that the matrix must be the size of the conjugate tranpose
of the standard matrix, i.e. numbers of rows and columns are swapped.
# Example
```julia
  A = RowMajorAdjointMatrix(zeros(Complex, ncols, nrows))

  io = open("./mmap.bin", "w+") do io
  Ammap = Mmap.mmap(io, Matrix{ComplexF64}, (ncols, nrows))
  A = RowMajorAdjointMatrix(Ammap)
```
"""
struct RowMajorAdjointMatrix{T,M<:AbstractArray{T}} <: AbstractArray{T, 2}
  matrix::M
end
function RowMajorAdjointMatrix(::Type{T}, rows, cols) where T
  return RowMajorAdjointMatrix(zeros(T, rows, cols))
end

function Base.setindex!(rmam::RowMajorAdjointMatrix, v, i, j)
  rmam.matrix[j, i] = v'
  return v
end
function Base.getindex(rmam::RowMajorAdjointMatrix, i, j)
  return rmam.matrix[j, i]'
end
function Base.view(rmam::RowMajorAdjointMatrix, i, j)
  return RowMajorAdjointMatrix((@view rmam.matrix[j, i]))
end

function LinearAlgebra.:\(rmam::RowMajorAdjointMatrix, b::AbstractVecOrMat)
  return (b' / rmam.matrix)'
end

struct LQRowMajorAdjointMatrix{LT,QT}
  L::LT
  Q::QT
end
struct LURowMajorAdjointMatrix{LT,UT,PT}
  L::LT
  U::UT
  P::PT
end

factorize!(rmam::RowMajorAdjointMatrix) = lq!(rmam)

for (op, st, m) in ((:lu!, :LURowMajorAdjointMatrix, :L),
                    (:lq!, :LQRowMajorAdjointMatrix, :L))
  @eval Base.eltype(rmam::$st) = eltype(rmam.$m)
  @eval function LinearAlgebra.$op(rmam::RowMajorAdjointMatrix)
    return $st($op(rmam.matrix)...)
  end
end
function LinearAlgebra.:\(rmam::LQRowMajorAdjointMatrix,
                          b::AbstractVecOrMat)
  x = zeros(promote_type(eltype(b), eltype(rmam)), length(b))
  mul!(x, rmam.Q, b)
  resize!(x, size(rmam.L, 1))
  ldiv!(LowerTriangular(rmam.L)', x)
  return x
#  return LowerTriangular(rmam.L)' \ mul(rmam.Q * b)[1:size(rmam.L, 1)]
end

function LinearAlgebra.:\(rmam::LURowMajorAdjointMatrix,
                          b::AbstractVecOrMat)
  return ((b' / rmam.U) / rmam.L)'[sortperm(rmam.P)]
end

Base.iterate(rmam::RowMajorAdjointMatrix) = iterate(rmam.matrix)
Base.iterate(rmam::RowMajorAdjointMatrix, s) = iterate(rmam.matrix, s)
Base.length(rmam::RowMajorAdjointMatrix) = length(rmam.matrix)
Base.size(rmam::RowMajorAdjointMatrix) = reverse(size(rmam.matrix))
Base.size(rmam::RowMajorAdjointMatrix, i) = size(rmam)[i]

Base.adjoint(rmam::RowMajorAdjointMatrix) = rmam.matrix
Base.:*(a, rmam::RowMajorAdjointMatrix) = a * rmam.matrix'
function Base.isapprox(a::AbstractArray, rmam::RowMajorAdjointMatrix{T,M}) where {T,M}
  return isapprox(a, rmam.matrix')
end
function Base.isapprox(rmam::RowMajorAdjointMatrix{T,M}, a::AbstractArray) where {T,M}
  return isapprox(a, rmam)
end

end
