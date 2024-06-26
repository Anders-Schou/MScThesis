# grep -r --exclude-dir=old "mse xx error" ./experiments/DoubleLaplace | sed 's/time_and_eval.dat.*//' | while read line
# do
#     python scripts/load_plot_and_eval.py --settings="${line}settings.json" --mainfilepath="${line}"
# done


# find . | grep settings.json

find ./experiments -type d -exec test -e '{}'/settings.json -a -e '{}'/main.py \; -exec python scripts/load_plot_and_eval.py --settings="{}/settings.json" --mainfilepath="{}" ';'