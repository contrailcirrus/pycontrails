C      
      SUBROUTINE DERIV(LAT,LON,ALT,TH,PH,RC,FL,
     &DJ,EM,H2O,M,O2,YP,Y,DTS,RO2)
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
      INTEGER NC,NR,NE,NIT,I,J,LAT,LON,ALT,TH,PH
      PARAMETER(NC=220,NR=606,NE=14,NIT=6)
      DOUBLE PRECISION,ALLOCATABLE :: EM(:,:,:,:)
      DOUBLE PRECISION,ALLOCATABLE RO2(:,:,:)
      DOUBLE PRECISION,ALLOCATABLE RC(:,:,:,:)
      DOUBLE PRECISION,ALLOCATABLE DJ(:,:,:,:)
      DOUBLE PRECISION,ALLOCATABLE H2O(:,:,:)
      DOUBLE PRECISION,ALLOCATABLE M(:)
      DOUBLE PRECISION,ALLOCATABLE O2(:,:,:)
      DOUBLE PRECISION,ALLOCATABLE Y(:,:,:,:)
      DOUBLE PRECISION,ALLOCATABLE YP(:,:,:,:)
      DOUBLE PRECISION,ALLOCATABLE FL(:,:,:,:)
      DOUBLE PRECISION,ALLOCATABLE P(:,:,:)
      DOUBLE PRECISION,ALLOCATABLE L(:,:,:)
      DOUBLE PRECISION DTS

      ALLOCATE(EM(LAT,LON,ALT,NC))
      ALLOCATE(RO2(LAT,LON,ALT))
      ALLOCATE(RC(LAT,LON,ALT,TH))
      ALLOCATE(DJ(LAT,LON,ALT,PH))
      ALLOCATE(H2O(LAT,LON,ALT))
      ALLOCATE(M(ALT))
      ALLOCATE(O2(LAT,LON,ALT))
      ALLOCATE(Y(LAT,LON,ALT,NC))
      ALLOCATE(YP(LAT,LON,ALT,NC))
      ALLOCATE(FL(LAT,LON,ALT,NR))
      ALLOCATE(P(LAT,LON,ALT))
      ALLOCATE(L(LAT,LON,ALT))

CF2PY INTENT(IN) :: LAT,LON,ALT,TH,PH,EM,RC,DJ,H2O,M,O2,DTS,YP
CF2PY INTENT(OUT) :: RO2
CF2PY INTENT(INOUT) :: Y,FL
      CHARACTER*6  CNAMES(NC)
C          O1D              Y(  1)
      P(:,:,:) = EM(:,:,:,1)
     &+(DJ(:,:,:,1)*Y(:,:,:,6))                                             
      L(:,:,:) = 0.0
     &+(RC(:,:,:,7))+(RC(:,:,:,8)) 
     &+(RC(:,:,:,16)*H2O(:,:,:))      
      Y(:,:,:,1) = P(:,:,:)/L(:,:,:)
      END  
      