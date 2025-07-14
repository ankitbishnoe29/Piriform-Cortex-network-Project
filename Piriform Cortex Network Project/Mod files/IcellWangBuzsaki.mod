TITLE Icellchann
: Cell model from Wang and Buzsaki, J Neurosci 1996
: Programmed by Adriano Tort, CBD, BU, 2008

UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	(nA) = (nanoamp)
}
 
NEURON {
	SUFFIX wangbuzaki
	NONSPECIFIC_CURRENT ina
	NONSPECIFIC_CURRENT ik
	NONSPECIFIC_CURRENT celldrive, celldrivenoise
	NONSPECIFIC_CURRENT il
	RANGE gna, gk, ena, ek, gl, el
	RANGE drive,drivenoise 
}
 
PARAMETER {
        v				(mV)
        celsius	= 33		(degC)
        gna= 0.035		(mho/cm2)
        ena= 55			(mV)
        gk= 0.009		(mho/cm2)
        ek= -90			(mV)
	gl= 0.0001		(mho/cm2)
        el= -65			(mV)
:		drive=-0.03		(nA/cm2)
:		drivenoise =0	(mA/cm2)
		}
 
STATE {
	m
	n
	h 
}
 
ASSIGNED {
	celldrive		(mA/cm2)
	celldrivenoise	(mA/cm2)
	
	ina		(mA/cm2) 
 	minf
	mtau   	(ms)
	hinf
	htau    (ms)
	
	ik		(mA/cm2)
	ninf
	ntau	(ms)
	
	il		(mA/cm2) 
}
 
BREAKPOINT {
        SOLVE states METHOD cnexp
		:celldrive = -drive
		:celldrivenoise = -drivenoise
		
        ina=gna*minf*minf*minf*h*(v-ena)
		ik=gk*n*n*n*n*(v-ek)
		il=gl*(v-el)      
}
 
DERIVATIVE states { 
       rates(v)
	   h'= 5*(hinf- h)/ htau
	   n'= 5*(ninf- n)/ ntau 
}


INITIAL { 
	rates(v)
	n=ninf
	h=hinf
	m=minf
	}

PROCEDURE rates(v (mV)) {
LOCAL alpha, beta, x, x1, x2
UNITSOFF 

	alpha = -0.1*(v+35)/(exp(-0.1*(v+35))-1)
	beta = 4*exp(-(v+60)/18)
	minf=alpha/(alpha+beta)
	
	alpha = 0.05*exp(-(v+58)/20)
	beta = 1/(exp(-0.1*(v+28))+1)
	htau=1/(alpha+beta)
	hinf=alpha*htau
	
	alpha = -0.01*(v+34)/(exp(-0.1*(v+34))-1)
	beta = 0.125*exp(-(v+44)/80)
	ntau=1/(alpha+beta)
	ninf=alpha*ntau	
	
UNITSON
}
