!edm
      subroutine euclidean_distance_matrix(x, y, nsamples, nfeat, r)

      implicit none
  
      double precision, dimension(:,:), intent(in) :: x
      double precision, dimension(:,:), intent(in) :: y
      integer, intent(in) :: nsamples
      integer, intent(in) :: nfeat
      double precision, dimension(:,:), intent(inout) :: r
  
      integer :: i, j, k
      double precision :: tmp
  
!$OMP PARALLEL DO PRIVATE(tmp)
      do i = 1, nsamples
          do j = 1, nsamples
              tmp = 0.0
              do k = 1, nfeat
                  tmp = (x(k,i) - y(k,j))**2 + tmp
              enddo
              r(i,j) = tmp
          enddo
      enddo
!$OMP END PARALLEL DO
  
      end subroutine euclidean_distance_matrix
