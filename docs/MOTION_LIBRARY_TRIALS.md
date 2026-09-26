# Source motion physics trials

All 61 recordings were tested individually at 1× speed with the bundled WBC policy.
54 completed; 7 stopped on a detected fall. There were no format or runtime errors.

Each trial starts from its own recorded initial pose. These trials do not establish
tracking quality at other speeds, arbitrary clip transitions, or real-hardware reliability.
Completion means no fall was detected; it does not certify accurate imitation.

Source: [exptech/g1-moves](https://huggingface.co/datasets/exptech/g1-moves/tree/895d064b2385725e6aabf540461c219afc3e0916) (CC BY 4.0).
Policy SHA-256: `2763b77e54f1618ed992057889d056250feda4667a2d56db654cd6c4a1e4ad8a`.

| Clip | Result | Simulated / total seconds | Joint RMSE (rad) |
| --- | --- | ---: | ---: |
| Dad Dance (`B_DadDance`) | completed | 41.80 / 41.80 | 0.085 |
| Long Dance (`B_LongDance`) | completed | 119.44 / 119.44 | 0.127 |
| Spiral Dance (`B_SpiralDance`) | completed | 47.78 / 47.78 | 0.101 |
| Stretch Dance (`B_StretchDance`) | completed | 43.08 / 43.08 | 0.128 |
| Wiggle Dance (`B_WiggleDance`) | completed | 37.26 / 37.26 | 0.083 |
| Step touch (`step_touch`) | completed | 32.48 / 32.48 | 0.101 |
| Gnarly (`J_Dance11_Gnarly`) | completed | 45.14 / 45.14 | 0.105 |
| Lush Life (`J_Dance12_LushLife`) | completed | 35.58 / 35.58 | 0.123 |
| Shuffle (`J_Dance17_Shuffle`) | completed | 32.02 / 32.02 | 0.136 |
| Tik Tok (`J_Dance18_TikTok`) | completed | 17.94 / 17.94 | 0.118 |
| Lets GO (`J_Dance19_LetsGO`) | completed | 24.84 / 24.84 | 0.111 |
| Modern (`J_Dance1_Modern`) | fallen | 17.62 / 37.74 | 0.152 |
| DWG (`J_Dance20_DWG`) | completed | 15.62 / 15.62 | 0.093 |
| Blunt (`J_Dance21_Blunt`) | completed | 28.74 / 28.74 | 0.107 |
| Thrilling (`J_Dance22_Thrilling`) | completed | 26.98 / 26.98 | 0.115 |
| Midnight Sun (`J_Dance23_MidnightSun`) | completed | 48.58 / 48.58 | 0.121 |
| Salsa (`J_Dance2_Salsa`) | fallen | 32.58 / 35.84 | 0.185 |
| Woah (`J_Dance3_Woah`) | completed | 29.98 / 29.98 | 0.100 |
| Broadway (`J_Dance4_Broadway`) | completed | 27.94 / 27.94 | 0.138 |
| Hype (`J_Dance5_Hype`) | fallen | 24.14 / 31.58 | 0.200 |
| Sassy (`J_Dance6_Sassy`) | completed | 31.88 / 31.88 | 0.117 |
| Party (`J_Dance7_Party`) | completed | 46.20 / 46.20 | 0.119 |
| West Coast (`J_Dance8_WestCoast`) | completed | 39.90 / 39.90 | 0.110 |
| Peace Maker (`J_Dance9_PeaceMaker`) | fallen | 43.92 / 58.34 | 0.132 |
| Single Ladies (`J_ShortDance13_SingleLadies`) | completed | 14.02 / 14.02 | 0.090 |
| Disco (`J_ShortDance14_Disco`) | completed | 14.28 / 14.28 | 0.110 |
| Nineties (`J_ShortDance15_Nineties`) | completed | 13.10 / 13.10 | 0.119 |
| Jazz walk (`walk`) | completed | 10.60 / 10.60 | 0.123 |
| Attack Karate (`B_AttackKarate`) | fallen | 39.96 / 47.18 | 0.124 |
| Karate bow (`bow`) | completed | 37.56 / 37.56 | 0.105 |
| Chops Karate (`B_ChopsKarate`) | completed | 52.66 / 52.66 | 0.119 |
| Crazy Chops Karate (`B_CrazyChopsKarate`) | completed | 24.18 / 24.18 | 0.119 |
| Forward Karate (`B_ForwardKarate`) | completed | 36.32 / 36.32 | 0.108 |
| Long Karate (`B_LongKarate`) | completed | 48.00 / 48.00 | 0.119 |
| Spin Karate (`B_SpinKarate`) | completed | 25.66 / 25.66 | 0.124 |
| Move 1 (`M_Move1`) | completed | 31.16 / 31.16 | 0.107 |
| Move 10 (`M_Move10`) | completed | 30.52 / 30.52 | 0.137 |
| Move 11 (`M_Move11`) | completed | 16.58 / 16.58 | 0.150 |
| Move 17 (`M_Move17`) | completed | 26.62 / 26.62 | 0.103 |
| Move 18 (`M_Move18`) | completed | 16.70 / 16.70 | 0.103 |
| Move 19 (`M_Move19`) | completed | 16.86 / 16.86 | 0.108 |
| Move 2 (`M_Move2`) | completed | 36.44 / 36.44 | 0.119 |
| Move 20 (`M_Move20`) | completed | 23.56 / 23.56 | 0.110 |
| Move 3 (`M_Move3`) | completed | 31.96 / 31.96 | 0.124 |
| Move 4 (`M_Move4`) | completed | 29.24 / 29.24 | 0.134 |
| Move 5 (`M_Move5`) | fallen | 31.54 / 35.98 | 0.138 |
| Move 6 (`M_Move6`) | completed | 35.60 / 35.60 | 0.110 |
| Move 7 (`M_Move7`) | completed | 23.80 / 23.80 | 0.121 |
| Move 8 (`M_Move8`) | fallen | 9.94 / 14.04 | 0.212 |
| Move 9 (`M_Move9`) | completed | 27.28 / 27.28 | 0.119 |
| Short Move 12 (`M_ShortMove12`) | completed | 7.78 / 7.78 | 0.106 |
| Short Move 13 (`M_ShortMove13`) | completed | 8.98 / 8.98 | 0.119 |
| Short Move 14 (`M_ShortMove14`) | completed | 6.54 / 6.54 | 0.091 |
| Short Move 15 (`M_ShortMove15`) | completed | 6.78 / 6.78 | 0.107 |
| Short Move 16 (`M_ShortMove16`) | completed | 7.38 / 7.38 | 0.168 |
| Fence 1 (`B_Fence1`) | completed | 26.98 / 26.98 | 0.091 |
| Fence 2 (`B_Fence2`) | completed | 10.62 / 10.62 | 0.129 |
| Hands Chop (`B_HandsChop`) | completed | 29.42 / 29.42 | 0.082 |
| Hands Up (`B_HandsUp`) | completed | 7.36 / 7.36 | 0.097 |
| Rocamena (`V_Rocamena`) | completed | 9.48 / 9.48 | 0.088 |
| Pull Over (`V_PullOver`) | completed | 42.50 / 42.50 | 0.136 |
