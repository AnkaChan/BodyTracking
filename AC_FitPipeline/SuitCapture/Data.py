import json
import numpy as np
import cv2

CIDList1487 = '''PU2 Q10
V62 VE0
UY1 V63
UU2 V50
GJ2
GJ1 GQ3
GG1 GP3
G72 GG0
G21 G73
G12 G70
UR2 V30
UQ1 UR3
GF2 GP0
G71 GF3
GF1 YT3
YC2 GF0
G11 YC3
YK2 YT0
YC1 YK3
UP2 UR0
Y42 YC0
YB2 YK0
Y41 YB3
UK2
YJ2 YR0
Y32 YB0
UK1
U41 UB3
TT2 U40
YA2 YJ0
Y31 YA3
UJ2
UB1 UJ3
U32 UB0
TT1 U33
TK2 TT0
YQ2
YG2 YQ0
YA1 YG3
Y22 YA0
UJ1
UA2 UJ0
U31 UA3
TR2 U30
TK1 TR3
TK0
YQ1
Y72 YG0
Y21 Y73
UG2
UA1 UG3
U22 UA0
TR1 U23
TJ2 TR0
TJ3
YP2
YG1 YP3
Y20
UG1
U72 UG0
U21 U73
TQ2 U20
TJ1 TQ3
TJ0
YY2
YP1 YY3
YF2 YP0
Y71 YF3
Y12 Y70
Y13
UF2
U71 UF3
U70 U12
TQ1 U13
TG2 TQ0
TG3
YY1
YM2 YY0
YF1 YM3
Y11 Y63
VT2 Y10
VT3
U62 UF0
U11 U63
TP2 U10
TG1 TP3
TG0
YV2
YM1 YV3
YE2 YM0
Y61 YE3
VY2 Y60
VT1 VY3
VT0
UF1
U61 UE3
TY2 U60
TP1 TY3
TF2 TP0
TF3
YV1
YL2 YV0
YE1 YL3
Y52 YE0
VY1 Y53
UE2
TM2 TY0
TF1 TM3
TF0
YU2
YD2 YL0
Y51 YD3
VU2 Y50
UM3
UM2
UM0
UM1
UL0
UL1
UL2
UL3
PY2 Q30
PV2 Q20
PV1 Q13
PF1 PM3
PE1 PL3
RD1 RK3
RD2 RL0
PD2 PL0
RC2 RK0
PK2
PD1 PK3
PJ2
PC1 PJ3
P32 PC0
RC1 RJ3
RB2 RJ0
PJ1
PB2 PJ0
P31 PB3
MY2 P30
RB1 QK3
RB0
MR2
PB1 MR3
MY1 MB3
MR1
MJ2 MR0
MB1 MJ3
MQ2
MJ1 MQ3
M31 MA3
M30
MQ1
MG2 MQ0
MA1 MG3
M22 MA0
RA2
MG1 MP3
M72 MG0
M21 M73
QF1 QP3
Q62 QF0
MM2
MF1 MM3
M62 MF0
M11 M63
LT2 M10
LT3
Q61 QE3
Q63
Q60
MV2
MM1 MV3
ME2 MM0
M61 ME3
LY2 M60
LT1 LY3
LT0
QE1 QM3
Q52 QE0
Q53
MV1
ML2 MV0
ME1 ML3
M52 ME0
LY1 M53
LR2 LY0
LR3
QD2 QM0
Q51 QD3
Q50
MU2
ML1 MU3
MD2 ML0
M51 MD3
LV2 M50
LR1 LV3
LR0
QD1 QL3
Q42 QD0
Q43
MT2
MK1 MT3
MC2 MK0
M41 MC3
LU2 M40
LP1 LU3
ER2 UD0
TV1 ER3
EE2 TV0
TE1 EE3
TE0
LM2
LF1 LM3
LB2 LF0
L41 LB3
G10
G13
G20
G22 GA0
GA2 GJ0
G30
G31 GA3
G32 GB0
G33
G40
G41 GB3
G42 GC0
G43
G50
G51 GC3
G52 GD0
G53
G60
G61 GD3
G62 GE0
G63
GB1 GJ3
GB2 GK0
GK1
GK2
GC1 GK3
GC2 GL0
GD1 GL3
GD2 GM0
GE1 GM3
GM1
GM2
GL1
GL2
GP1
GP2
GG2 GQ0
GQ1
GQ2
L42 YD0
VU1 L43
VR1 VU3
VR0
VR2 VY0
VR3
LF2 YU0
YD1 LF3
LP0
LP2 LV0
LP3
LV1 M43
M42 MD0
M12 M70
M13
M20
M23
M32 MB0
M33
M71 MF3
MA2 MJ0
MB2 PB0
MD1 MK3
MF2 MP0
MK2 MU0
MP1
MP2
MU1
MY3
MY0
P10
P11 P33
P12 P40
P13
P20
P21 P43
P22 P50
P23
P41 PC3
P42 PD0
P51 PD3
P52 PE0
P53
P60
P61 PE3
P62 PF0
P63
P70
P71 PF3
P72 PG0
P73
PC2 PK0
PE2 PM0
PF2 PQ0
PG1 PQ3
PG2 PR0
PA1 PG3
PK1
PL1
PL2
PM1
PM2
PQ1
PQ2
PR2
PR1
PU1 PR3
PA2 PU0
PT1 PU3
PT2 PV0
PV3
PY0
PY1 Q23
PY3
Q12
Q11
Q21
Q22
Q31
Q32
Q33
Q70
Q71 QF3
Q72 QG0
Q73
QA0
QA1 QG3
QA2 QJ0
QA3
QC0
QC1 QJ3
QC2 QK0
QC3
QF2 QQ0
QG1 QQ3
QG2 QT0
QJ1 QT3
QJ2 QU0
QK1 QU3
QK2 RQ0
QL2 QV0
QM1 QV3
QM2 QY0
QP1 QY3
QP2 R10
QQ1 R13
QQ2 R20
QT1 R23
QT2 R30
QU1 R33
QU2 R40
QV1 R53
QV2 R60
QY1 R63
R12 RA0
R11 R73
R21 RA3
R22
R31
R32
R41
R42
RQ1 R43
R51
R52
R61
R62
QY2 R70
R71
R72
RA1
RB3
RC0
RC3
RD0
RD3
RE0
RE1 RL3
RE2 RM0
RE3
RF0
RF1 RM3
RF2 RP0
RF3
RG0
RG1 RP3
RG3
RJ1 RQ3
RJ2 RR0
RK1 RR3
RK2 RU0
RU3 RL1
RL2 RV0
RM1 RV3
RM2 RY0
RV2
RY1
RY2
RP1 RY3
RP2 T10
T11
T12
RT1 T13
RQ2 T20
RR1 T23
RR2 T30
T21
T22
T31
T32
RU1 T33
RU2 T40
T41
T42
T43 RV1
TE2 TM0
TE3
TK3
TL0
TL1 TT3
TL2 TU0
TL3
TM1 TV3
TU1 U43
TU3 UP1
TV2 U50
TY1 U53
U42 UC0
U51 UD3
U52 UE0
UB2 UK0
UC1 UK3
UC2
V21 UC3
UD2
UE1
UP0
UP3
UQ0
UQ2 UT0
UQ3
UT1 V33
TU2 V20
UT3
UU0
UU1 V43
UU3
UV0
UV1 V53
UV2 V60
UV3
UY0
UY2 V70
UY3
V73 V11
V10
V12 VK0
V13
V22 VA0
V23 UR1
V31 VA3
V32 VB0
UT2 V40
V41 VB3
V42 VC0
V51 VC3
V52 VD0
V61 VD3
V71 VE3
V72 VF0
VA1
VA2
VB1
VB2
VC1
VC2
VD1
VD2
VE1
VE2
VF1
VF2
VK1 VF3
VG0
VG1 VK3
VG3
VP0 VK2
VP1
VP2
VL1 VP3
VG2 VL0
VL2 VQ0
VL3 VJ1
VM0 VJ2
VM1 VQ3
VM2
VM3
VJ0
VJ3
VQ1
VQ2
Y23
Y30
Y33
Y40
Y43
Y62 YF0
PA0
PA3
YB1 YJ3
YJ1 YQ3
YK1 YR3
YL1 YU3
YR1
YR2
YT1
YT2
YU1
G23
GA1 GG3
PT0
PT3
QE2 QP0
DJ2 DM0
1A1
DM1 1P3
1A2
1B1
1E2
D72 1K0
1J1
1F1
1F2
D61
D62 1J0
1Q2 160
1K1 113
1P2 150
151 1E3
171
1A3
E32
E33
DY2 E30
E31 1T3
1T2 1A0
E22 1T0
1T1 173
DV2 E20
E21 1R3
1R2 170
172
1R1 163
162
E12 1R0
DV1 E13
DR2 DV0
DR1 DU3
DU2 E10
E11 1Q3
161 1F3
DD1 DF3
DF2 DU0
DU1 DM3
DC2 DF0
DF1 DJ3
DC3
152 1F0
1Q1 153
DM2 1Q0
DE2 DJ0
DC1 DE3
DC0
1E1
142 1E0
1P1 143
DL2 1P0
DJ1 DL3
DE1 DG3
DB2 DE0
DB3
1D2
141 1D3
1M2 140
DL1 1M3
DG2 DL0
D22 DG0
DB1 D23
DB0
1D1
132 1D0
1M1 133
DK2 1M0
DG1 DK3
D21 D53
1C2
131 1C3
1L2 130
DK1 1L3
D52 DK0
D12 D50
D11 D43
CU2 D10
CU3
1G2
1G1
1C1 1G3
122 1C0
1L1 123
DA2 1L0
D51 DA3
CY2 D40
CU1 CY3
121 1B3
1K2 120
DA1 1K3
D42 DA0
CY1 D33
D32 D70
D71 1J3
1J2 110
111
112 1B0
1B2 1G0
BR2
BR3
CQ2
CQ3
CR2
CR3
BR1 BT3
CT2 CQ0
BT2
BL2 BT0
CT1 CP3
BL1 BP3
CK2 CP0
CF2 CK0
BK2 BP0
BG1 BK3
CM2 BG0
CK1 CM3
CF0
CF1 CJ3
CJ2 CM0
CM1 BF3
BF2 BK0
CL2 BF0
CJ1 CL3
CE2 CJ0
CE3
CD2 CL0
CE1 CD3
CE0
CL1 B33
CD1 CA3
CA2 B30
CC2 CA0
CC1 C73
C72 B20
C71 B13
CB2 C70
C11 CB3
CB1 C63
BY2 CB0
BY3
C62 B10
B11 B43
BV2 C30
C31 C53
AR2 AU0
C22 C50
C51 AM3
AM2 AR0
AR1 AT3
C42 AM0
AM1 AQ3
AQ2 AT0
AT1
AL0
AL1
AL2 AQ0
AL3 C41
C40
C52 AP0
AP1 AR3
AP2 B40
AP3 C61
B41 AU3
AU1 AV3
AU2 BC0
BC1 AY3
BC2
AQ1
AT2 AV0
AV2 AY0
AY1
AY2
B12 B50
B21 B53
B22 B60
B23 CA1
B31 B63
B32 BJ0
B51 B73
B71 BC3
B72 BD0
B52 BA0
BA1 BD3
BA2 BE0
B61 BA3
B62 BB0
BB1 BE3
BB2
BJ1 BB3
BJ2 BM0
BM1
BM2
BK1 BM3
BD1
BD2
BE1
BE2
BF1 BJ3
BG2 BL0
BG3 CP1
BL3 BQ1
BP1
BP2
CP2 BQ0
BQ2 BR0
CQ1 BQ3
BT1
BU0
BU1 BV3
BU2 BY0
BU3
BV0
BV1 C23
BY1 C33
C20
C21 C43
C10
C12 CC0
C13
C32 C60
QB1 CC3
QB0
QB2 CD0
QB3
CF3
CG0
CG1 CK3
CG2 CT0
CG3
CR0
CR1 CT3
CU0
CV0
CV1 D13
CV2 D20
CV3
CY0
D30
D31 D63
D60
D41 D73
DD0
DD3
DP0
DP1 DR3
DP2 DT0
DP3
DQ0
DQ1 DT3
DT1 DV3
DT2 DY0
DY1 E23
QR1 DY3
DQ2 QR0
QR2
QR3
AV1
B42 B70
DD2 DR0
DQ3
EY2
ER1 EY3
EK2 ER0
EE1 EK3
E72 EE0
E73
LM1
LQ2 LM0
LQ3 LB1
L32 LB0
KC2
K61 KC3
JY2 K60
JR1 JY3
JJ2 JR0
JD1 JJ3
JD0
MT1
J62 MT0
MC1 J63
FT2 MC0
LU1 FT3
FF2 LU0
FF3
UD1
EK1 EQ3
ED2 EK0
E71 ED3
E70
LK2
LK3 LQ1
LA2 LQ0
LA3 L31
KU2 L30
KM1 KU3
KG2 KM0
KC1 KG3
K52 KC0
JY1 K53
JQ2 JY0
JJ1 JQ3
JC2 JJ0
JC3
JA1
J52 JA0
J11 J53
FR2 J10
FL1 FR3
FE2 FL0
FA1 FE3
F42 FA0
EY1 F43
EQ2 EY0
EJ2 EQ0
ED1 EJ3
E62 ED0
E63
LK1
LE2 LK0
LA1 LE3
L22 LA0
KU1 L23
KL2 KU0
KG1 KL3
KB2 KG0
K51 KB3
JV2 K50
JQ1 JV3
JG2 JQ0
JC1 JG3
JC0
J72
J51 J73
FY2 J50
FR1 FY3
FK2 FR0
FE1 FK3
F72 FE0
F41 F73
EV2 F40
EQ1 EV3
EJ1 EP3
EC2 EJ0
E61 EC3
E60
LJ2
LJ3 LE1
L72 LE0
L21 L73
KT2 L20
KL1 KT3
KF2 KL0
KB1 KF3
K42 KB0
JV1 K43
JP2 JV0
JG1 JP3
JB2 JG0
JB3
J71
J42 J70
FY1 J43
FQ2 FY0
FK1 FQ3
FD2 FK0
F71 FD3
F32 F70
EV1 F33
EP2 EV0
EG2 EP0
EC1 EG3
E52 EC0
E53
LD2 LJ0
L71 LD3
L12 L70
KT1 L13
KK2 KT0
KF1 KK3
KA2 KF0
K41 KA3
JU2 K40
JP1 JU3
JF2 JP0
JB1 JF3
JB0
J41
FV2 J40
FQ1 FV3
FJ2 FQ0
FD1 FJ3
F62 FD0
F31 F63
EU2 F30
EP1 EU3
EG1 EM3
EB2 EG0
E50
LG2
LD1 LG3
L62 LD0
L11 L63
KR2 L10
KK1 KR3
KE2 KK0
KA1 KE3
K32 KA0
JU1 K33
JM2 JU0
JF1 JM3
1Y2 JF0
1Y3
J32
FV1 J33
FP2 FV0
FJ1 FP3
FC2 FJ0
F61 FC3
F22 F60
EU1 F23
EM2 EU0
EF2 EM0
EB1 EF3
E42 EB0
E43
LG1
LC2 LG0
L61 LC3
KY2 L60
KR1 KY3
KJ2 KR0
KE1 KJ3
K72 KE0
K31 K73
JT2 K30
JM1 JT3
JE2 JM0
1Y1 JE3
1Y0
J31
FU2 J30
FP1 FU3
FG2 FP0
FC1 FG3
F52 FC0
F21 F53
ET2 F20
EM1 ET3
EF1 EL3
EA2 EF0
E41 EA3
E40
LC1
L52 LC0
KY1 L53
KQ2 KY0
KJ1 KQ3
KD2 KJ0
K71 KD3
K22 K70
JT1 K23
JL2 JT0
JE1 JL3
JE0
J22
FU1 J23
FU0 FM2
FG1 FM3
FB2 FG0
F51 FB3
F12 F50
ET1 F13
EL2 ET0
6C2 EL0
EA1 6C3
5Q2 EA0
5Q3
562
L51 563
4K2 L50
KQ1 4K3
412 KQ0
KD1 413
3E2 KD0
K21 3E3
2U2 K20
JL1 2U3
2A2 JL0
2A3
AB2 J20
FM1 AB3
7P2 FM0
FB1 7P3
FB0
F11
6Y2 F10
EL1 6Y3
6C1 6L3
622 6C0
5Q1 623
5Q0
561
4U2 560
4K1 4U3
4A2 4K0
411 4A3
3P2 410
3E1 3P3
352 3E0
2U1 353
2J2 2U0
2A1 2J3
2A0
AK2
AB1 AK3
A12 AB0
7P1 A13
7D2 7P0
7D3
6Y1
6L2 6Y0
6B2 6L0
621 6B3
5P2 620
5P3
552
4U1 553
4J2 4U0
4A1 4J3
3Y2 4A0
3P1 3Y3
3D2 3P0
351 3D3
2T2 350
2J1 2T3
272 2J0
273
AK1
AA2 AK0
A11 AA3
7M2 A10
7D1 7M3
7D0
6V2
6L1 6V3
612 6B0
5P1 613
5P0
551
4T2 550
4J1 4T3
472 4J0
3Y1 473
3M2 3Y0
3D1 3M3
342 3D0
2T1 343
2G2 2T0
271 2G3
270
AJ2
AA1 AJ3
7Y2 AA0
7M1 7Y3
7C2 7M0
7C3
6V1
6K2 6V0
6B1 6K3
611 6A3
5M2 610
5M3
542
4T1 543
4G2 4T0
471 4G3
3V2 470
3M1 3V3
3C2 3M0
341 3C3
2R2 340
2G1 2R3
262 2G0
263
AJ1
A72 AJ0
7Y1 A73
7L2 7Y0
7C1 7L3
7C0
6U2
6K1 6U3
6A2 6K0
5Y2 6A0
5M1 5Y3
5M0
5D2
541 5D3
4R2 540
4G1 4R3
462 4G0
3V1 463
3L2 3V0
3C1 3L3
332 3C0
2R1 333
2F2 2R0
261 2F3
260
AG2
A71 AG3
7V2 A70
7B2 7L0
7B3
6U1
6J2 6U0
6A1 6J3
5Y1 673
5L2 5Y0
5L3
5D1
532 5D0
4R1 533
4F2 4R0
461 4F3
3U2 460
3L1 3U3
3B2 3L0
331 3B3
2Q2 330
2F1 2Q3
252 2F0
253
AG1
A62 AG0
7V1 A63
7K2 7V0
7B1 7K3
7B0
6T2
6J1 6T3
672 6J0
5L1 5V3
5L0
5C2
531 5C3
4Q2 530
4F1 4Q3
452 4F0
3U1 453
3K2 3U0
3B1 3K3
322 3B0
2Q1 323
2E2 2Q0
251 2E3
1V2 250
1V3
AF2
A61 AF3
7U2 A60
7K1 7U3
7A2 7K0
7A3
6T1
6G2 6T0
671 6G3
5V2 670
5K2 5V0
5K3
5C1
522 5C0
4Q1 523
4E2 4Q0
451 4E3
3T2 450
3K1 3T3
3A2 3K0
321 3A3
2P2 320
2E1 2P3
242 2E0
1V1 243
1V0
AE2
A51 AE3
7T2 A50
7J1 7T3
772 7J0
741 773
740
6R1
6F2 6R0
661 6F3
5U2 660
5K1 5U3
5E2 5K0
5B2
521 5B3
4P2 520
4E1 4P3
442 4E0
3T1 443
3J2 3T0
3A1 3J3
312 3A0
2P1 313
2D2 2P0
241 2D3
1U2 240
AE1
A42 AE0
7T1 A43
7G2 7T0
771 7G3
732 770
733
6Q2
6F1 6Q3
652 6F0
5U1 653
5J2 5U0
5E1 5J3
5B1
512 5B0
4P1 513
4D2 4P0
441 4D3
3R2 440
3J1 3R3
372 3J0
311 373
2M2 310
2D1 2M3
232 2D0
1U1 233
1U0
A41 AD3
7R2 A40
7G1 7R3
762 7G0
731 763
730
6Q1
6E2 6Q0
651 6E3
5T2 650
5J1 5T3
5J0
5A2
4M2 510
4D1 4M3
432 4D0
3R1 433
3G2 3R0
371 3G3
2Y2 370
2M1 2Y3
2C2 2M0
231 2C3
230
AD2
A32 AD0
7R1 A33
7F2 7R0
761 7F3
722 760
723
6P2
6E1 6P3
642 6E0
5T1 643
5G2 5T0
5G3
5A1
4Y2 5A0
4M1 4Y3
4C2 4M0
431 4C3
3Q2 430
3G1 3Q3
362 3G0
2Y1 363
2L2 2Y0
2C1 2L3
222 2C0
223
AC2
A31 AC3
7Q2 A30
7F1 7Q3
752 7F0
721 753
720
6P1
6D2 6P0
641 6D3
5R2 640
5G1 5R3
5G0
4L2 4Y0
4C1 4L3
422 4C0
3Q1 423
3F2 3Q0
361 3F3
2V2 360
2L1 2V3
2B2 2L0
221 2B3
220
A22 AC0
7Q1 A23
7E2 7Q0
751 7E3
712 750
713
6M2
6D1 6M3
632 6D0
5R1 633
5F2 5R0
5F3
4B2 4L0
421 4B3
420
3F1
3F0
2V1
2K2 2V0
2B1 2K3
212 2B0
213
A21
A20
7E1
7E0
711
710
6M1
6M0
631
630
4B1
2K1
2K0
211
210
1U3
4B0
4V3 4L1
4V0
4V1
4V2 570
4Y1 573
571
572
511 5A3
5E0
5E3
5F0
5F1
5V1 663
662 6G0
6G1 6R3
6R2
742 7A0
743
7A1 7J3
7J2 7U0
7U1 A53
A52 AF0
AC1
AD1
E51 EB3
FA2 FF0
FA3
FF1 FL3
FL2 FT0
FT1 J13
J12 J60
J21
J61 JA3
JA2
JD2 JK0
JD3
JK1 JR3
JK2 QL0
Q41 JK3
JR2 K10
K11 K63
K12 R50
QL1 K13
K62
KM2 KV0
KM3
KP0
KP1 KV3
KP2 VU0
KP3
KV1 L33
KV2 L40
LJ1
Q40
7L1 7V3
AF1
TA2 TD0
TD1
TD2
TD3
T62 TA0
T63
T61 T73
T60
TA1 TC3
TA3
TC2
TC1
T72 TC0
T71 TB3
T70 T52
T50
T51 RT3
RG2 RT0
RT2 TB0
TB1
TB2
GT0
GT1 GV3
GT2 GY0
GT3
GY1
GY2
GY3
GV1
GV2
GU0 GE2
GU1
GU2
GU3 GR1
GR3
GR0
GR2 GV0
T53
GE3'''


codeDictOldWithStar = ['1', '2', '3', '4', '5', '6', '7', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'K', 'L',
                               'M', 'P', 'Q', 'R', 'T', 'U', 'V', 'Y', '*']

characterToDigit = {character:iC for iC, character in enumerate(codeDictOldWithStar)}

def readProcessJsonFile(file):
    data = json.load(open(file))

    corners = data['corners']

    codes = data['code']
    accetped_qi = data['accept_qi']
    numCodes = len(codes)

    cornerKeys = [[] for i in range(len(corners))]
    cornerKeysConfidence = [[] for i in range(len(corners))]
    for iQuad in range(numCodes):
        code = codes[iQuad]
        qi = accetped_qi[iQuad]

        quadIndex = 0
        for iC in qi:
            cornerKeys[iC].append(code+str(quadIndex))
            quadIndex += 1
            probablity1 = np.exp(data['recog_dens4a'][iQuad])/np.sum(np.exp(data['recog_dens4a'][iQuad]))
            probablity2 = np.exp(data['recog_dens4b'][iQuad])/np.sum(np.exp(data['recog_dens4b'][iQuad]))

            characterDigit1 = characterToDigit[code[0]]
            characterDigit2 = characterToDigit[code[1]]
            cornerKeysConfidence[iC].append(probablity1[characterDigit1]*probablity2[characterDigit2])

    return cornerKeys, cornerKeysConfidence





class CornerLabeler:
    def __init__(s, CIDList=CIDList1487):
        s.labelingFunction = {}
        allCornerKeys = CIDList.split('\n')
        s.numCorners = 0
        for cId, cornerKey in enumerate(allCornerKeys):
            if cornerKey != '':
                s.numCorners += 1
                singleKeys = cornerKey.split(' ')
                for singleKey in singleKeys:
                    s.labelingFunction[singleKey] = cId

    def labelCorners(s, cornerkeys, consistencyCheckScheme='discard', cornerKeysConfidence=None):
        cornerIds = []
        cornerConf = []
        for iC, cornerKey in enumerate(cornerkeys):
            if len(cornerKey) == 1:
                cornerIds.append(s.labelingFunction.get(cornerKey[0], -1))
                if cornerKeysConfidence is not None and consistencyCheckScheme == 'maxConfidence':
                    cornerConf.append(cornerKeysConfidence[iC][0])
            elif len(cornerKey) == 2:
                if consistencyCheckScheme == 'discard':
                    uId1 = s.labelingFunction.get(cornerKey[0], -1)
                    uId2 = s.labelingFunction.get(cornerKey[1], -1)

                    if uId1 == uId2:
                        cornerIds.append(uId1)
                    else:
                        cornerIds.append(-1)

                elif consistencyCheckScheme == 'maxConfidence':
                    uId1 = s.labelingFunction.get(cornerKey[0], -1)
                    uId2 = s.labelingFunction.get(cornerKey[1], -1)

                    if cornerKeysConfidence[iC][0] > cornerKeysConfidence[iC][1]:
                        cornerIds.append(uId1)
                        cornerConf.append(cornerKeysConfidence[iC][0] )
                    else:
                        cornerIds.append(uId2)
                        cornerConf.append(cornerKeysConfidence[iC][1])

            else:
                cornerIds.append(-1)
                cornerConf.append(0)

        if consistencyCheckScheme == 'discard':
            return cornerIds
        elif consistencyCheckScheme == 'maxConfidence':
            return cornerIds, cornerConf

    def cornerUIdsToCorrList(s, corners, cornerUIds, cornerConf=None):
        corrs = [[-1,-1] for i in range(s.numCorners)]
        corrConfs = [0 for i in range(s.numCorners)]
        if cornerConf is None:
            for iC, corner in enumerate(corners):
                if cornerUIds[iC] != -1:
                    corrs[cornerUIds[iC]] = corner
        else:
            for iC, corner in enumerate(corners):
                uId = cornerUIds[iC]
                if uId != -1:
                    if corrs[uId][0] == -1:
                        corrs[uId] = corner
                        corrConfs[uId] = cornerConf[iC]
                    else:
                        if corrConfs[uId] < cornerConf[iC]:
                            corrs[uId] = corner
                            corrConfs[uId] = cornerConf[iC]

        return corrs





class CornerLabelData:
    def __init__(self, labelFile, imgFile, cropSize = 50, checkSingleLabeledCorners = True, inCornerData=None):
        self.cropSize = cropSize
        self.img = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
        self.imgPadded = cv2.copyMakeBorder(self.img, cropSize, cropSize, cropSize,
                                            cropSize, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        self.imgPadded = self.imgPadded[:,:,np.newaxis]
        self.cornersData = []
        if not labelFile == '':
            self.cornersData.extend(json.load(open(labelFile)))

        if inCornerData is not None:
            self.cornersData.extend(inCornerData)

        self.cornersToCheck = []
        if checkSingleLabeledCorners:
            for ic, c in enumerate(self.cornersData):
                # assert self.cornersData[ic]['code'][0] != '0' or self.cornersData[ic]['code'][1] != '0'
                if self.cornersData[ic]['code'][0] == '0' or self.cornersData[ic]['code'][1] == '0':
                    self.cornersToCheck.append(ic)


    def getCornerCrop(self, cId):
        c = self.cornersData[cId]
        cp = c['coord']
        cxi = int(cp[0]) if cp[0] - np.floor(cp[0]) <= 0.5 else int(cp[0]) + 1
        cyi = int(cp[1]) if cp[1] - np.floor(cp[1]) <= 0.5 else int(cp[1]) + 1

        neighborhood = self.imgPadded[cyi:cyi + 2 * self.cropSize + 1, cxi:cxi + 2 * self.cropSize + 1, :]

        return neighborhood, c

    def checkData(self, cId):
        cIdAll = self.cornersToCheck[cId]
        c = self.cornersData[cIdAll]
        cp = c['coord']
        cxi = int(cp[0]) if cp[0] - np.floor(cp[0]) <= 0.5 else int(cp[0]) + 1
        cyi = int(cp[1]) if cp[1] - np.floor(cp[1]) <= 0.5 else int(cp[1]) + 1

        neighborhood = self.imgPadded[cyi:cyi + 2 * self.cropSize + 1, cxi:cxi + 2 * self.cropSize + 1, :]

        return neighborhood, c

    def saveCornerData(self, outJsonFile):
        json.dump(self.cornersData, open(outJsonFile, 'w'))

def getAvailibaleCodeSet4Code(CIDList = CIDList1487):
    cornerCodes = CIDList.split('\n')
    # print(len(cornerCodes))

    oldIndexToNew = [3, 2, 0, 1]

    codeSet = []
    for i in range(len(cornerCodes)):
        # skip empty line
        if len(cornerCodes[i])<3:
            continue
        code4 = ['0', '0', '0', '0']
        code2 = cornerCodes[i].split(' ')
        if len(code2) == 2:
            assert (code2[0][2] == '0' and code2[1][2] == '2') or \
                   (code2[0][2] == '2' and code2[1][2] == '0') or \
                   (code2[0][2] == '1' and code2[1][2] == '3') or \
                   (code2[0][2] == '3' and code2[1][2] == '1')
            code4[oldIndexToNew[int(code2[0][2])]] = code2[0][:2]
            code4[oldIndexToNew[int(code2[1][2])]] = code2[1][:2]
        else:
            code4[oldIndexToNew[int(code2[0][2])]] = code2[0][:2]
        codeSet.append(code4)
    return codeSet

def getCodeSet2(CIDList = CIDList1487):
    cornerCodes = CIDList.split('\n')
    codeSet = []
    for i in range(len(cornerCodes)):
        if len(cornerCodes[i]) >= 3:
            code2 = cornerCodes[i].split(' ')
            codeSet.append({'Code': [c[:2] for c in code2], 'Id': [int(c[2:]) for c in code2]})

    return codeSet

if __name__ == '__main__':
    labeler = CornerLabeler()