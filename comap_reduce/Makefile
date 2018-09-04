FCOMP=gfortran

INC=-I/Path/to/slalib/include
LIB=-L/Path/to/slalib/lib


LIBS=-lsla 
FLAGS=

all: EphemNew 

EphemNew:

	f2py -c --f90exec=$(FCOMP) $(FLAGS) -m EphemNew EphemNew.f90 $(INC) $(LIB) $(LIBS)
