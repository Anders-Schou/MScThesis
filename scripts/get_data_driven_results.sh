filenames=("Small" "Only_data_small" "SemiSmall" "Only_data_semismall" "Medium" "Only_data_medium" "Large" "Only_data")
methodnames=("10" "10*" "100" "100*" "1000" "1000*" "10000" "10000*")
stresses=("$\stress{xx}$" "$\shearstress{xy}$" "$\stress{yy}$" "$\stress{\text{v}}$")
stressnames=("xx" "xy" "yy" "vm_stress")

echo -n "Amount of data " > $1
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
    grep -Po "L2rel ${stressnames[$k]} error: \K.*" experiments/DoubleLaplace/DoubleNet/Basic/logs/DoubleNet/eval_error.dat | tr -d '\n' >> $1
    echo -n "$" >> $1
done
echo "\\\\ \hline" >> $1

for j in 0 1 2 3 4 5 6 7
do
    for k in 0 1 2 3
    do
        if [[ $k == 0 ]]
        then
            echo -n "${methodnames[$j]} " >> $1
        fi
        echo -n " & $" >> $1
        grep -Po "L2rel ${stressnames[$k]} error: \K.*" experiments/DoubleLaplace/DoubleNet/Data/logs/${filenames[$j]}/eval_error.dat | tr -d '\n' >> $1
        echo -n "$" >> $1
    done
    echo "\\\\ \hline" >> $1
done
