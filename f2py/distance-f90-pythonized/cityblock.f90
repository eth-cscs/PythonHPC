!cityblock
      subroutine cityblock_distance_matrix(x, y, nsamples, nfeat, r)

      implicit none
  
      double precision, intent(in) :: x(nfeat, nsamples)
      double precision, intent(in) :: y(nfeat, nsamples)
      integer, intent(in) :: nsamples
      integer, intent(in) :: nfeat
      double precision, intent(out) :: r(nsamples, nsamples)
  
      integer :: i, j, k
      double precision :: tmp
  
!$OMP PARALLEL DO PRIVATE (tmp)
      do i = 1, nsamples
          do j = 1, nsamples
              tmp = 0.0
              do k = 1, nfeat
                  tmp = abs(x(k,i) - y(k,j)) + tmp
              enddo
              r(i,j) = tmp
          enddo
      enddo
!$OMP END PARALLEL DO
  
      end subroutine cityblock_distance_matrix
