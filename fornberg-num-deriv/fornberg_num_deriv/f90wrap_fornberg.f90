! Module fornberg_deriv defined in file fornberg.f90
subroutine fornberg_grid_derivative(iorder, iaccuracy, grid, u, du, n)
    use fornberg_deriv, only: grid_derivative
    implicit none
    
    integer, intent(in) :: iorder
    integer, intent(in) :: iaccuracy
    real(8), intent(in), dimension(n) :: grid
    real(8), intent(in), dimension(n) :: u
    real(8), intent(inout), dimension(n) :: du
    integer :: n
    !f2py intent(hide), depend(grid) :: n = shape(grid,0) 
    call grid_derivative(iorder=iorder, iaccuracy=iaccuracy, grid=grid, ngrid=n, u=u, du=du)
end subroutine fornberg_grid_derivative

! End of module fornberg_deriv defined in file fornberg.f90

