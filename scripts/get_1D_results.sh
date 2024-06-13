if [ $1 ]
then
    echo "Sin" > $1
    for inner_dir in 100 1000 10000
    do
        echo -n "${inner_dir} collocation points" >> $1
        for outer_dir in Data PINN1 PINN2 PINN3 PINN4 PINN5
        do
            echo -n " & " >> $1
            grep -Po 'Converged to a tol of 0.01 in \K.* epochs' experiments/1D/Sin/logs/$outer_dir/$inner_dir/time_and_eval.dat | tr -st ' \n' '$' | tr -d 'epochs\n' | sed 's/^/$/' >> $1
            grep -Po 'Did not converge to a tol of 0.01 in \K.* epochs' experiments/1D/Sin/logs/$outer_dir/$inner_dir/time_and_eval.dat | tr -st ' \n' '$' | tr -d 'epochs\n' | sed 's/^/$>/' >> $1 
        done
        echo "\\\\" >> $1
    done
    echo >> $1
    echo "ExpSin" >> $1

    for inner_dir in 100 1000 10000
    do
        echo -n "${inner_dir} collocation points" >> $1    
        for outer_dir in Data PINN1 PINN2 PINN3 PINN4 PINN5
        do
            echo -n " & " >> $1
            grep -Po 'Converged to a tol of 0.01 in \K.* epochs' experiments/1D/ExpSin/logs/$outer_dir/$inner_dir/time_and_eval.dat | tr -st ' \n' '$' | tr -d 'epochs\n' | sed 's/^/$/' >> $1
            grep -Po 'Did not converge to a tol of 0.01 in \K.* epochs' experiments/1D/ExpSin/logs/$outer_dir/$inner_dir/time_and_eval.dat | tr -st ' \n' '$' | tr -d 'epochs\n' | sed 's/^/$>/' >> $1 
        done
        echo "\\\\" >> $1
    done
fi

