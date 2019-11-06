!mm
      subroutine matrix_multiplication(A, B, n, C)

      implicit none
        
      double precision, intent(in) :: A(n, n)
      double precision, intent(in) :: B(n, n)
      integer, intent(in) :: n
      double precision, intent(out) :: C(n, n)

      integer :: i, j

      !$OMP PARALLEL DO
      do i = 1, n 
          do j = 1, n
              C(j,i) = sum(A(:,j) * B(i,:))
          enddo
      enddo
      !$OMP END PARALLEL DO

      end subroutine matrix_multiplication
