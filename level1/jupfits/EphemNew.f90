!!Quick routine to wrap the SLALIB moon position function.
!!It is inaccurate but fine for QUIJOTE accuracy. It downsamples
!! the data by 100x!

subroutine rdplan(jd, np, lon, lat, ra, dec, diam, len)
  implicit none
  
  integer, intent(in) :: len
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  integer, intent(in) :: np
  real*8, intent(in) :: jd(len)
  real*8, intent(out) :: diam(len)
  real*8, intent(out) :: ra(len)
  real*8, intent(out) :: dec(len)

  !f2py integer len
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,jd, diam

  !integer :: i

  real*8 :: pi = 3.14159265359

  integer :: step = 100
  integer :: kup,k
  kup = len/step


  do k=1, len
     call sla_rdplan(jd(k),np,lon*pi/180.0,lat*pi/180.0,ra(k),dec(k),diam(k))
  enddo
  
end subroutine rdplan




subroutine planet(jd, np, dist, len)
  implicit none
  
  integer, intent(in) :: len
  integer, intent(in) :: np
  real*8, intent(in) :: jd(len)
  real*8, intent(out) :: dist(6,len)

  !f2py integer len
  !f2py real*8 jd, dist

  integer :: jstat

  integer :: step = 100
  integer :: kup,k
  kup = len/step

  !mask = 1.0d0

  do k=1, len
     call sla_planet(jd(k),np,dist(:,k),jstat)
  enddo
  
end subroutine planet


subroutine h2e(az, el, mjd, lon, lat, ra, dec, len)
  implicit none
  
  integer, intent(in) :: len
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: az(len)
  real*8, intent(in) :: el(len)
  real*8, intent(in) :: mjd(len)
  real*8, intent(out) :: ra(len)
  real*8, intent(out) :: dec(len)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface


  interface
     real*8 FUNCTION sla_dranrm(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_dranrm
  end interface

  !f2py integer len
  !f2py real*8 lon, lat
  !f2py real*8 ra,dec,mjd

  integer :: i

  real*8 :: gmst

  do i=1, len
     call sla_dh2e(az(i), el(i), lat, ra(i), dec(i))
     gmst = sla_gmst(mjd(i))
     ra(i) = gmst + lon - ra(i)
     ra(i) = sla_dranrm(ra(i))
  enddo    

  
end subroutine h2e

subroutine precess(ra, dec,mjd, len)
  implicit none
  
  integer, intent(in) :: len
  real*8, intent(in) :: mjd(len)
  real*8, intent(inout) :: ra(len)
  real*8, intent(inout) :: dec(len)

  interface
     real*8 FUNCTION sla_epb(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_epb
  end interface

  !f2py integer len
  !f2py real*8 mjd
  !f2py real*8 ra,dec

  integer :: i
  real*8 :: epoch

  do i=1, len
     epoch = sla_epb(mjd(i))
     call sla_preces('FK5', epoch, 2000D0, ra(i), dec(i))
  enddo    

  
end subroutine precess

subroutine pa(ra, dec,mjd, lon,lat,pang, len)
  implicit none
  
  integer, intent(in) :: len
  real*8, intent(in) :: lon
  real*8, intent(in) :: lat
  real*8, intent(in) :: mjd(len)
  real*8, intent(in) :: ra(len)
  real*8, intent(in) :: dec(len)
  real*8, intent(out) :: pang(len)

  interface
     real*8 FUNCTION sla_gmst(mjddummy)
     real*8 :: mjddummy
     END FUNCTION sla_gmst
  end interface

  interface
     real*8 FUNCTION sla_pa(radummy, decdummy, phidummy)
     real*8 :: radummy, decdummy, phidummy
     END FUNCTION sla_pa
  end interface

  !f2py integer len
  !f2py real*8 mjd, lon, lat
  !f2py real*8 ra,dec, pang

  integer :: i
  real*8 :: ha, gmst

  do i=1, len
     gmst = sla_gmst(mjd(i))
     ha = gmst + lon - ra(i)
     pang(i) = sla_pa(ha, dec(i), lat)
  enddo    

  
end subroutine pa


subroutine e2g(ra, dec, gl, gb, len)
  implicit none
  
  integer, intent(in) :: len
  real*8, intent(in) :: ra(len)
  real*8, intent(in) :: dec(len)
  real*8, intent(out) :: gl(len)
  real*8, intent(out) :: gb(len)

  !f2py integer len
  !f2py real*8 ra,dec,gl,gb

  integer :: i


  do i=1, len
     call sla_eqgal(ra(i), dec(i), gl(i), gb(i))
  enddo    

  
end subroutine e2g
