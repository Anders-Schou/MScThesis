layers=(1 2 4)
layernames=("Shallow" "Medium" "Deep")
neurons=(64 128 256)
neuronnames=("_and_small" "_and_medium" "_and_large")
stresses=("$\stress{xx}$" "$\shearstress{xy}$" "$\stress{yy}$")
stressnames=("xx" "xy" "yy")

echo -n "Layers & Neurons " > $1
for i in 0 1 2
do
    echo -n "& ${stresses[$i]} " >> $1
done
echo "\\\\ \hline" >> $1

for i in 0 1 2
do
    for j in 0 1 2
    do
        for k in 0 1 2
        do
            if [[ $j == 0 && $k == 0 ]]
            then
                echo -n "\multirow{3}{*}{${layers[$i]}} " >> $1
            fi
            if [[ $k == 0 ]]
            then
                echo -n "& ${neurons[$j]}" >> $1
            fi
            echo -n " & $" >> $1
            grep -Po "L2rel ${stressnames[$k]} error: \K.*" experiments/Biharmonic/Tweaking/Width_and_depth/logs/${layernames[$i]}${neuronnames[$j]}/time_and_eval.dat | tr -d '\n' >> $1
            echo -n "$" >> $1
        done
        echo "\\\\" >> $1
    done
    echo "\hline" >> $1
done