# Check if an environment is already loaded
if [[ "$VIRTUAL_ENV" != "" ]]
then
    echo -e "\nDeactivated current environment [${VIRTUAL_ENV}]\n"
    deactivate
fi

# Load python
module purge
module load python3/3.12.1

# Activate virtual environment
. ~/.venv/venv_MSc/bin/activate
