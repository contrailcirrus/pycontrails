C      
      SUBROUTINE DERIV(RC,FL,DJ,EM,H2O,M,O2,YP,Y,DTS,RO2)
C-----------------------------------------------------------------------
C     PURPOSE:    -  TO EVALUATE CONCENTRATIONS Y FROM RATE COEFFICIENTS
C                    J VALUES AND EMISSION RATES.
C                    DETAILED CHEMISTRY
C
C     INPUTS:     -  SPECIES CONCENTRATIONS (Y), RATE COEFFICIENTS (RC),
C                    PHOTOLYSIS RATES (DJ), AND EMISSIONS (EM).
C
C     OUTPUTS:    -  CONCENTRATIONS FROM CHEMISTRY AND EMISSIONS (DY).
C
C
C     CREATED:    -  1-SEPT-1994   Colin Johnson
C
C     VER 2:      -  20-FEV-1995   Colin Johnson
C                                  New chemical scheme with 50 species
C                                  from chem3.txt mechanism.
C
C     VER NEW:    -  25-NOV-1996  Colin Johnson using Dick's new scheme
C                                  with 70 species.
C                     2-DEC-1996  Dick Derwent edited scheme
C-----------------------------------------------------------------------
      IMPLICIT NONE
C-----------------------------------------------------------------------
      INTEGER NC,NR,NE,NIT,I,J,LAT,LON,ALT
      PARAMETER(LAT=2,LON=2,ALT=2,NC=220,NR=606,NE=14,NIT=6)
      DOUBLE PRECISION EM(LAT,LON,ALT,NC)
      DOUBLE PRECISION RO2(LAT,LON,ALT)
      DOUBLE PRECISION RC(LAT,LON,ALT,510)
      DOUBLE PRECISION DJ(LAT,LON,ALT,96)
      DOUBLE PRECISION H2O(LAT,LON,ALT)
      DOUBLE PRECISION M(ALT)
      DOUBLE PRECISION O2(LAT,LON,ALT)
      DOUBLE PRECISION DTS
      DOUBLE PRECISION Y(LAT,LON,ALT,NC)
      DOUBLE PRECISION YP(LAT,LON,ALT,NC)
      DOUBLE PRECISION FL(LAT,LON,ALT,NR)
      DOUBLE PRECISION P(LAT,LON,ALT),L(LAT,LON,ALT)
CF2PY INTENT(IN) :: EM, RC, DJ, H2O, M, O2, DTS, YP
CF2PY INTENT(OUT) :: RO2
CF2PY INTENT(IN,OUT) :: Y, FL
      CHARACTER*6  CNAMES(NC)
C          O1D              Y(  1)
      P(:,:,:) = EM(:,:,:,1)
     &+(DJ(:,:,:,1)*Y(:,:,:,6))                                             
      L(:,:,:) = 0.0
     &+(RC(:,:,:,7))+(RC(:,:,:,8)) 
     &+(RC(:,:,:,16)*H2O(:,:,:))      
      Y(:,:,:,1) = P(:,:,:)/L(:,:,:)
      END  
      