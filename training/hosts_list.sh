#!/bin/bash

# ensipc200-219
# HOSTS=(ensipc200.ensimag.fr ensipc201.ensimag.fr ensipc202.ensimag.fr ensipc203.ensimag.fr ensipc204.ensimag.fr ensipc205.ensimag.fr ensipc206.ensimag.fr ensipc207.ensimag.fr ensipc208.ensimag.fr ensipc209.ensimag.fr ensipc210.ensimag.fr ensipc211.ensimag.fr ensipc212.ensimag.fr ensipc213.ensimag.fr ensipc214.ensimag.fr ensipc215.ensimag.fr ensipc216.ensimag.fr ensipc217.ensimag.fr ensipc218.ensimag.fr ensipc219.ensimag.fr)

# vmgpu01-50
HOSTS=(vmgpu001.ensimag.fr vmgpu002.ensimag.fr vmgpu003.ensimag.fr vmgpu004.ensimag.fr vmgpu005.ensimag.fr vmgpu006.ensimag.fr vmgpu007.ensimag.fr vmgpu008.ensimag.fr vmgpu009.ensimag.fr vmgpu010.ensimag.fr vmgpu011.ensimag.fr vmgpu012.ensimag.fr vmgpu013.ensimag.fr vmgpu014.ensimag.fr vmgpu015.ensimag.fr vmgpu016.ensimag.fr vmgpu017.ensimag.fr vmgpu018.ensimag.fr vmgpu019.ensimag.fr vmgpu020.ensimag.fr vmgpu021.ensimag.fr vmgpu022.ensimag.fr vmgpu023.ensimag.fr vmgpu024.ensimag.fr vmgpu025.ensimag.fr vmgpu026.ensimag.fr vmgpu027.ensimag.fr vmgpu028.ensimag.fr vmgpu029.ensimag.fr vmgpu030.ensimag.fr vmgpu031.ensimag.fr vmgpu032.ensimag.fr vmgpu033.ensimag.fr vmgpu034.ensimag.fr vmgpu035.ensimag.fr vmgpu036.ensimag.fr vmgpu037.ensimag.fr vmgpu038.ensimag.fr vmgpu039.ensimag.fr vmgpu040.ensimag.fr vmgpu041.ensimag.fr vmgpu042.ensimag.fr vmgpu043.ensimag.fr vmgpu044.ensimag.fr vmgpu045.ensimag.fr vmgpu046.ensimag.fr vmgpu047.ensimag.fr vmgpu048.ensimag.fr vmgpu049.ensimag.fr vmgpu050.ensimag.fr)

# keep only the first 10
HOSTS=("${HOSTS[@]:0:10}")
