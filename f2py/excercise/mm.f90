!mm
      subroutine matrix_multiplication('Add the arguments')

      implicit none
        
      'Define the arguments'

      integer :: i, j

      !$OMP PARALLEL DO
      do i = 1, n 
          do j = 1, n
              C(j,i) = sum(A(:,j) * B(i,:))
          enddo
      enddo
      !$OMP END PARALLEL DO

      end subroutine matrix_multiplication
