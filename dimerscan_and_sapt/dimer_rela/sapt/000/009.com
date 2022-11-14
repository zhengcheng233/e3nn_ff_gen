! shift= 1.150000
memory,1.4,G
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
17,C,,17.85344315,17.26756287,15.51650047
18,H,,18.57444191,18.02356339,15.58250046
19,H,,17.24844170,17.17356300,16.39750099
20,O,,18.39244270,16.03956413,15.12950039
21,C,,19.15444183,15.51856422,16.16850090
22,H,,19.44944191,14.59256363,15.72650051
23,H,,18.55744171,15.26156425,17.04250145
24,H,,20.01044273,16.20156479,16.49250031
25,C,,16.80244255,17.58356285,14.44849968
26,H,,15.94444275,17.02756310,14.70149994
27,H,,17.19844246,17.43456459,13.46850014
28,O,,16.31844330,18.87556458,14.66950035
29,C,,15.46744251,19.32756424,13.58549976
30,H,,15.24544239,20.38756371,13.86649990
31,H,,14.51344299,18.80656433,13.73649979
32,H,,15.89544296,19.12656403,12.57050037
33,He,,13.81013948,16.18430064,16.12378369
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

!=========delta(HF) contribution for higher order interaction terms====
ca=2101.2; cb=2102.2 !sapt files

!dimer
dummy,33
{df-hf,basis=jkfit,locorb=0}
edm=energy

!monomer A
dummy,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33

{df-hf,basis=jkfit,locorb=0; save,$ca}
ema=energy
{sapt;monomerA}

!monomer B
dummy,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,33

{df-hf,basis=jkfit,locorb=0; save,$cb}
emb=energy
{sapt;monomerB}

!interaction contributions
{sapt,SAPT_LEVEL=2;intermol,ca=$ca,cb=$cb,icpks=1,fitlevel=3
dfit,basis_coul=jkfit,basis_exch=jkfit,cfit_scf=3}

!calculate high-order terms by subtracting 1st+2nd order energies
eint_hf=(edm-ema-emb)*1000 mH
delta_hf=eint_hf-e1pol-e1ex-e2ind-e2exind

!=========DFT-SAPT at second order intermol. perturbation theory====
ca=2103.2; cb=2104.2 !sapt files;

!shifts for asymptotic correction to xc potential
eps_homo_PBE0_B=-0.139377
eps_homo_PBE0_A=-0.139377
ip_B=0.220086
ip_A=0.220086
shift_B=ip_B+eps_homo_pbe0_B !shift for bulk xc potential (B)
shift_A=ip_A+eps_homo_pbe0_A !shift for bulk xc potential (A)

!monomer A, perform LPBE0AC calculation
dummy,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33

{df-ks,pbex,pw91c,lhf,df_basis=jkfit; dftfac,0.75,1.0,0.25; asymp,shift_A; save,$ca}
{sapt;monomerA}

!monomer B, perform LPBE0AC calculation
dummy,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,33

{df-ks,pbex,pw91c,lhf,df_basis=jkfit; dftfac,0.75,1.0,0.25; start,atdens; asymp,shift_B; save,$cb}
{sapt;monomerB}

!interaction contributions
{sapt,SAPT_LEVEL=3;intermol,ca=$ca,cb=$cb,icpks=0,fitlevel=3,nlexfac=0.0
dfit,basis_coul=jkfit,basis_exch=jkfit,basis_mp2=mp2fit,cfit_scf=3}

!add high-order approximation to obtain the total interaction energy
eint_dftsapt=e12tot+delta_hf

