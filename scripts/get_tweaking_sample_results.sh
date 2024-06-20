filenames=("BC_Small" "BC_SemiSmall" "BC_Medium" "BC_Large")
circ_points=("1000" "1000" "1500" "1500")
rect_points=("1000" "2000" "2000" "3000")
stresses=("$\stress{xx}$" "$\shearstress{xy}$" "$\stress{yy}$")
stressnames=("xx" "xy" "yy")

echo -n "Circle points & Rectangle points " > $1
for i in 0 1 2
do
    echo -n "& ${stresses[$i]} " >> $1
done
echo "\\\\ \hline" >> $1

for j in 0 1 2 3
do
    for k in 0 1 2
    do
        if [[ $k == 0 ]]
        then
            echo -n "${circ_points[$j]} & ${rect_points[$j]}" >> $1
        fi
        echo -n " & $" >> $1
        grep -Po "L2rel ${stressnames[$k]} error: \K.*" experiments/Biharmonic/Tweaking/Sample_size/BC/logs/${filenames[$j]}/time_and_eval.dat | tr -d '\n' >> $1
        echo -n "$" >> $1
    done
    echo "\\\\ \hline" >> $1
done
