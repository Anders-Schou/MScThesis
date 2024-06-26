filenames=("tanh" "silu" "sigmoid" "sin" "wavelet")
methodnames=("$\tanh(z)$" "$\text{SiLU}$" "$\sigma(z)$" "$\sin(z)$" "$\text{Wavelet}$")
stresses=("$\stress{xx}$" "$\shearstress{xy}$" "$\stress{yy}$" "$\stress{\text{v}}$")
stressnames=("xx" "xy" "yy" "vm_stress")

echo -n "Activation function " > $1
for i in 0 1 2 3
do
    echo -n "& ${stresses[$i]} " >> $1
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
        grep -Po "L2rel ${stressnames[$k]} error: \K.*" experiments/Biharmonic/Tweaking/Activation/logs/${filenames[$j]}/eval_error.dat | tr -d '\n' >> $1
        echo -n "$" >> $1
    done
    echo "\\\\ \hline" >> $1
done
