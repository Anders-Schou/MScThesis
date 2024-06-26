filenames=("SingleNet" "SingleNet_extra_dense" "SingleNet_extra_layers")
methodnames=("2 Layers with 256 neurons" "2 Layers with 256 neurons*" "4 Layers with 256 neurons")
stresses=("$\stress{xx}$" "$\shearstress{xy}$" "$\stress{yy}$" "$\stress{\text{v}}$")
stressnames=("xx" "xy" "yy" "vm_stress")

echo -n "Configuration " > $1
for i in 0 1 2 3
do
    echo -n "& ${stresses[$i]} " >> $1
done
echo "\\\\ \hline" >> $1

for k in 0 1 2 3
do
    if [[ $k == 0 ]]
    then
        echo -n "Baseline " >> $1
    fi
    echo -n " & $" >> $1
    grep -Po "L2rel ${stressnames[$k]} error: \K.*" experiments/Biharmonic/Tweaking/Initialization/logs/He_normal/eval_error.dat | tr -d '\n' >> $1
    echo -n "$" >> $1
done
echo "\\\\ \hline" >> $1


for j in 0 1 2
do
    for k in 0 1 2 3
    do
        if [[ $k == 0 ]]
        then
            echo -n "${methodnames[$j]} " >> $1
        fi
        echo -n " & $" >> $1
        grep -Po "L2rel ${stressnames[$k]} error: \K.*" experiments/DoubleLaplace/SingleNet/Basic/logs/${filenames[$j]}/eval_error.dat | tr -d '\n' >> $1
        echo -n "$" >> $1
    done
    echo "\\\\ \hline" >> $1
done




echo >> $1
echo >> $1



filenames=("DoubleNet" "DoubleNet_extra_dense" "DoubleNet_extra_layers" "DoubleNet_half_size" "DoubleNet_half_size_double_width")
methodnames=("2 Layers with 256 neurons" "2 Layers with 256 neurons*" "4 Layers with 256 neurons" "2 Layers with 128 neurons" "4 Layers with 128 neurons")
stresses=("$\stress{xx}$" "$\shearstress{xy}$" "$\stress{yy}$" "$\stress{\text{v}}$")
stressnames=("xx" "xy" "yy" "vm_stress")

echo -n "Configuration " >> $1
for i in 0 1 2 3
do
    echo -n "& ${stresses[$i]} " >> $1
done
echo "\\\\ \hline" >> $1


for k in 0 1 2 3
do
    if [[ $k == 0 ]]
    then
        echo -n "Baseline " >> $1
    fi
    echo -n " & $" >> $1
    grep -Po "L2rel ${stressnames[$k]} error: \K.*" experiments/Biharmonic/Tweaking/Initialization/logs/He_normal/eval_error.dat | tr -d '\n' >> $1
    echo -n "$" >> $1
done
echo "\\\\ \hline" >> $1

for j in 0 1 2 3 4
do
    for k in 0 1 2 3
    do
        if [[ $k == 0 ]]
        then
            echo -n "${methodnames[$j]} " >> $1
        fi
        echo -n " & $" >> $1
        grep -Po "L2rel ${stressnames[$k]} error: \K.*" experiments/DoubleLaplace/DoubleNet/Basic/logs/${filenames[$j]}/eval_error.dat | tr -d '\n' >> $1
        echo -n "$" >> $1
    done
    echo "\\\\ \hline" >> $1
done
