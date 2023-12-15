module fornberg_deriv
    implicit none
    ! define parameters used in the subroutines
    integer, parameter :: mmax=8, nmax=20, iaccuracy_max=20
    
contains

    subroutine fdcoef(mord,nord,x0,grid,coef)
        integer, intent(in) :: mord, nord
        real(8), intent(in) :: x0
        real(8), intent(in), dimension(nord) :: grid
        real(8), intent(out), dimension(nord) :: coef

        integer :: nu, nn, mm, nmmin
        real(8) :: c1, c2, c3, c4, pfac
        real(8), dimension(mmax,nmax,nmax) :: weight

        weight = 0.0d0
        weight(1,1,1) = 1.0d0
        c1 = 1.0d0
        nmmin = min(nord,mord)

        do nn = 2,nord
            c2 = 1.0d0
            do nu=1,nn-1
                c3 = grid(nn) - grid(nu)
                c2 = c2 * c3
                c4 = 1.0d0 / c3
                pfac = grid(nn) - x0
                weight(1,nn,nu) = c4 * ( pfac * weight(1,nn-1,nu) )
                do mm = 2,nmmin
                    weight(mm,nn,nu) = c4 * ( pfac * weight(mm,nn-1,nu) - dfloat(mm-1) * weight(mm-1,nn-1,nu) )
                enddo
            enddo
            pfac = (grid(nn-1) - x0)
            weight(1,nn,nn) = ( c1 / c2 ) * ( -pfac * weight(1,nn-1,nn-1) )
            c4 = ( c1 / c2 )
            do mm = 2,nmmin
                weight(mm,nn,nn) = c4 * ( dfloat(mm-1) * weight(mm-1,nn-1,nn-1) - pfac * weight(mm, nn-1, nn-1) )
            enddo
            c1 = c2
        enddo
        ! store the coefficients and return
        do nu = 1,nord
            coef(nu) = weight(mord,nord,nu)
        enddo
    end subroutine fdcoef

    subroutine grid_derivative(iorder,iaccuracy,grid,ngrid,u,du)
        integer, intent(in) :: iorder, iaccuracy, ngrid
        real(8), intent(in), dimension(ngrid) :: grid, u
        real(8), intent(out), dimension(ngrid) :: du
        integer :: m, n, i, imid, jend
        real(8), dimension(iaccuracy_max) :: coef
        ! Check if requested accuracy is reasonable
        if (iaccuracy .gt. iaccuracy_max) stop 'requested accuracy order too large in routine'
        ! Declare aliases
        m = iorder + 1
        n = iaccuracy + 1
        ! Retrieve parameters that handle boundaries and differ for even or odd order accuracy
        if ( mod(iaccuracy, 2) .eq. 0 ) then
            imid = iaccuracy / 2
            jend = imid 
        else
            imid = ( iaccuracy / 2 ) + 1
            jend = iaccuracy / 2
        endif
        ! Handle the starting points of the grid
        do i = 1,imid 
            call fdcoef(m,n,grid(i),grid(1),coef)
            du(i) = dot_product(coef(1:n), u(1:n))
        enddo
        ! Handle the middle points of the grid
        do i = imid+1,ngrid-jend
            call fdcoef(m,n,grid(i),grid(i-imid),coef)
            du(i) = dot_product(coef(1:n), u(i-imid:i+jend))
        enddo
        ! Handle the end points of the grid
        do i = ngrid-jend+1,ngrid
            call fdcoef(m,n,grid(i),grid(ngrid-iaccuracy),coef)
            du(i) = dot_product(coef(1:n), u(ngrid-iaccuracy:ngrid))
        enddo        
    end subroutine grid_derivative
end module fornberg_deriv