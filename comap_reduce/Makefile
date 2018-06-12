QUIJOTEFCOMP=ifort
QUIJOTEINC=-I/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/Mapping/lib/include -I/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/Mapping/lib/include
QUIJOTELIB=-L/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/Mapping/lib/lib -L/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/Mapping/lib/lib -L/scratch/nas_falcon/scratch/sharper/QUIJOTE/Pipeline/DataAccess/lib

LIBS=-lsla -lhealpix -lcfitsio
FLAGS=

all: EphemNew 

EphemNew:

	f2py -c --f90exec=$(QUIJOTEFCOMP) $(FLAGS) -m EphemNew EphemNew.f90 $(QUIJOTEINC) $(QUIJOTELIB) $(LIBS)
