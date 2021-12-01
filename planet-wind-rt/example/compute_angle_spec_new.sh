SPEC="python /Users/morganmacleod/Dropbox/PlanetWind/Analysis/planet-wind-rt/Planet_Wind_Ray_Star_Origin24.py --base_dir ./ --level 4 --N_radial 100 --f_raystep 0.05 --angles  -0.3 -0.29 -0.28 -0.27 -0.26 -0.25 -0.24 -0.23 -0.22 -0.21 -0.2 -0.19 -0.18 -0.17 -0.16 -0.15 -0.14 -0.13 -0.12 -0.11 -0.1 -0.09 -0.08 -0.07 -0.06 -0.05 -0.04 -0.03 -0.02 -0.01 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 --snapshot PW_W107.out1.00030.athdf --scale 0.2 --savedata"
SPECp="python /Users/morganmacleod/Dropbox/PlanetWind/Analysis/planet-wind-rt/Planet_Wind_Ray_Star_Origin24.py --base_dir ./ --level 4 --N_radial 100 --f_raystep 0.05 --angles  -0.3 -0.29 -0.28 -0.27 -0.26 -0.25 -0.24 -0.23 -0.22 -0.21 -0.2 -0.19 -0.18 -0.17 -0.16 -0.15 -0.14 -0.13 -0.12 -0.11 -0.1 -0.09 -0.08 -0.07 -0.06 -0.05 -0.04 -0.03 -0.02 -0.01 0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 --snapshot PW_W107.out1.00030.athdf --scale 0.2 --savedata --parker --mdot 9103870660.468765"

cd A; $SPECp; $SPEC
#cd A; $SPEC
cd ../B; $SPEC
cd ../C; $SPEC

#cd A; $SPECa; $SPECb; $SPECc; $SPECd; $SPECe
#cd ../B;  $SPECa; $SPECb; $SPECc; $SPECd; $SPECe
#cd C;  $SPECa; $SPECb; $SPECc; $SPECd; $SPECe
