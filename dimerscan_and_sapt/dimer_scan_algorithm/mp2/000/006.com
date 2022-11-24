! shift = -0.366667
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
17,C,,16.49313545,16.78412437,15.98139191
18,H,,17.21413422,17.54012489,16.04739189
19,H,,15.88813305,16.69012451,16.86239052
20,O,,17.03213501,15.55612469,15.59439182
21,C,,17.79413414,15.03512478,16.63339043
22,H,,18.08913422,14.10912418,16.19139099
23,H,,17.19713402,14.77812481,17.50739098
24,H,,18.65013504,15.71812439,16.95738983
25,C,,15.44213390,17.10012436,14.91339111
26,H,,14.58413410,16.54412460,15.16639137
27,H,,15.83813381,16.95112610,13.93339157
28,O,,14.95813370,18.39212608,15.13439178
29,C,,14.10713387,18.84412575,14.05039120
30,H,,13.88513374,19.90412521,14.33139133
31,H,,13.15313435,18.32312584,14.20139122
32,H,,14.53513432,18.64312553,13.03539181
33,He,,13.12998530,15.94258123,16.35622924
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
