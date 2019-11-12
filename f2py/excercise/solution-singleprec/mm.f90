!mm
      subroutine matrix_multiplication(A, B, n, C)

      implicit none
        
      real, intent(in) :: A(n, n)
      real, intent(in) :: B(n, n)
      real, intent(out) :: C(n, n)
      integer, intent(in) :: n

      integer :: i, j

      !$OMP PARALLEL DO
      do i = 1, n 
          do j = 1, n
              C(j,i) = sum(A(:,j) * B(i,:))
          enddo
      enddo
      !$OMP END PARALLEL DO

      end subroutine matrix_multiplication
