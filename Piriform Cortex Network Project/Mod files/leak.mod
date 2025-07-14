TITLE passive leak current

NEURON {
	SUFFIX leak
        NONSPECIFIC_CURRENT i
	RANGE  i, e, g
}

PARAMETER {
	g = 0.0003 (mho/cm2) <0, 1e9>
        e = -75 (mV)
}
UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
	(um) = (micron)
} 

ASSIGNED {
	i   (mA/cm2)
        v    (mV)				
}

BREAKPOINT {
        i=g*(v-e)
} 



