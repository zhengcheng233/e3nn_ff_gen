! shift = 2.450000
memory,800,m
gdirect; gthresh,energy=1.d-8,orbital=1.d-8,grid=1.d-8
spherical
angstrom
symmetry,nosym
orient,noorient
geometry={
1,C,,10.58300018,15.66199970,17.44899940
2,H,,10.76099968,16.22100067,16.48900032
3,H,,9.84799957,16.25000000,17.99600029
4,O,,11.84200001,15.37500000,18.01399994
5,C,,12.57900047,16.51499939,18.41099930
6,H,,12.73999977,17.28700066,17.68400002
7,H,,13.55000019,16.03400040,18.56999969
8,H,,12.10599995,16.95999908,19.32200050
9,C,,10.02200031,14.35099983,17.06200027
10,H,,10.10400009,13.69600010,17.93300056
11,H,,10.56000042,13.85799980,16.32099915
12,O,,8.68500042,14.51799965,16.58799934
13,C,,8.05599976,13.22000027,16.43199921
14,H,,7.06099987,13.30700016,16.05400085
15,H,,7.92700005,12.69200039,17.36199951
16,H,,8.51399994,12.56700039,15.66800022
17,C,,19.01942253,17.68194008,15.11802197
18,H,,19.74042130,18.43794060,15.18402195
19,H,,18.41442108,17.58794022,15.99902153
20,O,,19.55842209,16.45394135,14.73102188
21,C,,20.32042122,15.93294144,15.77002144
22,H,,20.61542130,15.00694084,15.32802200
23,H,,19.72342110,15.67594147,16.64402199
24,H,,21.17642212,16.61594009,16.09402084
25,C,,17.96842194,17.99794006,14.05002117
26,H,,17.11042213,17.44194031,14.30302143
27,H,,18.36442184,17.84894180,13.07002163
28,O,,17.48442078,19.28994179,14.27102184
29,C,,16.63342094,19.74194145,13.18702126
30,H,,16.41142082,20.80194092,13.46802139
31,H,,15.67942238,19.22094154,13.33802128
32,H,,17.06142235,19.54094124,12.17202187
33,He,,14.39312893,16.39148924,15.92454436
}
basis={
set,orbital
default=avtz         !for orbitals
s,He,even,nprim=5,ratio=2.5,center=0.5
p,He,even,nprim=5,ratio=2.5,center=0.5
d,He,even,nprim=3,ratio=2.5,center=0.3
f,He,even,nprim=2,ratio=2.5,center=0.3
set,jkfit
default=avtz/jkfit   !for JK integrals
s,He,even,nprim=5,ratio=2.5,center=0.5
p,He,even,nprim=5,ratio=2.5,center=0.5
d,He,even,nprim=3,ratio=2.5,center=0.3
f,He,even,nprim=2,ratio=2.5,center=0.3
set,mp2fit 
default=avtz/mp2fit  !for E2disp/E2exch-disp
s,He,even,nprim=5,ratio=2.5,center=0.5
p,He,even,nprim=5,ratio=2.5,center=0.5
d,He,even,nprim=3,ratio=2.5,center=0.3
f,He,even,nprim=2,ratio=2.5,center=0.3
set,dflhf
default=avtz/jkfit   !for LHF
s,He,even,nprim=5,ratio=2.5,center=0.5
p,He,even,nprim=5,ratio=2.5,center=0.5
d,He,even,nprim=3,ratio=2.5,center=0.3
f,He,even,nprim=2,ratio=2.5,center=0.3
}

!dimer
dummy,33
df-hf, df_basis=jkfit
df-mp2, df_basis=mp2fit
edm=energy

!monomer A
dummy,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33

df-hf, df_basis=jkfit
df-mp2, df_basis=mp2fit
ema=energy
{sapt;monomerA}

!monomer B
dummy,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,33

df-hf, df_basis=jkfit
df-mp2, df_basis=mp2fit
emb=energy

eint=edm-ema-emb
