TITLE na3
: Na current for axon. No slow inact.
: M.Migliore Jul. 1997

NEURON {
	SUFFIX nax
	USEION na READ ena WRITE ina
	RANGE  gbar
	GLOBAL minf, hinf, mtau, htau,thinf, qinf
}

PARAMETER {
	gbar = 0.020   	(mho/cm2)	
								
	tha  =  -38.9	(mV)		: v 1/2 for act	
	qa   = 6.5	(mV)		: act slope (4.5)		
	Ra   = 0.4	(/ms)		: open (v)		
	Rb   = 0.124 	(/ms)		: close (v)		

	thi1 = -55	(mV)		: v 1/2 for inact 	
	thi2  = -55	(mV)		: v 1/2 for inact 	
	qd   = 1.3	(mV)	        : inact tau slope
	qg   = 1.3        (mV)
	mmin = 0.05	
	hmin = 1.95			
	q10 = 2.5
	Rg   = 0.02 	(/ms)		: inact recov (v) 	
	Rd   = 0.01 	(/ms)		: inact (v)	

	thinf  = -55 	(mV)		: inact inf slope	
	qinf  = 2 	(mV)		: inact inf slope 

	ena = 50	(mV)            : must be explicitly def. in hoc
	celsius 
	v 		(mV)
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

ASSIGNED {
	ina 		(mA/cm2)
	thegna		(mho/cm2)
	minf 		hinf 		
	mtau (ms)	htau (ms) 	
}
 

STATE { m h}

BREAKPOINT {
        SOLVE states METHOD cnexp
        thegna = gbar*m*m*m*h
	ina = thegna * (v - ena)
} 

INITIAL {
	trates(v)
	m=minf  
	h=hinf
}

DERIVATIVE states {   
        trates(v)      
        m' = (minf-m)/mtau
        h' = (hinf-h)/htau
}

PROCEDURE trates(vm) {  
        LOCAL  a, b, qt
        qt=q10^((celsius-24)/10)
	a = trap0(vm,tha,Ra,qa)
	b = trap0(-vm,-tha,Rb,qa)
	mtau = 1/(a+b)/qt
        if (mtau<mmin) {mtau=mmin}
	minf = a/(a+b)

	a = trap0(vm,thi1,Rd,qd)
	b = trap0(-vm,-thi2,Rg,qg)
	htau =  1/(a+b)/qt
        if (htau<hmin) {htau=hmin}
	hinf = 1/(1+exp((vm-thinf)/qinf))
}

FUNCTION trap0(v,th,a,q) {
	if (fabs(v-th) > 1e-6) {
	        trap0 = a * (v - th) / (1 - exp(-(v - th)/q))
	} else {
	        trap0 = a * q
 	}
}	

        

