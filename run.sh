if [ $1 ]
then
    python experiments/$1/main.py --settings="experiments/$1/settings.json" 
else
    python main.py --settings="settings.json" 
fi