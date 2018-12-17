FCOMP=gfortran

# Include directories for SLALIB
INC=-I/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/Mapping/lib/include -I/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/Mapping/lib/include

# SLALIB  library paths
LIB=-L/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/Mapping/lib/lib -L/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/Mapping/lib/lib -L/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/DataAccess/lib

LINKS=-lsla 
FLAGS=

all: EphemNew

EphemNew:
	f2py -c --f90exec=$(FCOMP) $(FLAGS) -m EphemNew EphemNew.f90 $(INC) $(LIB) $(LINKS)
