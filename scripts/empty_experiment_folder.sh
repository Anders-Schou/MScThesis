if [ $1 ]
then
    if [ $2 ]
    then
        rm -rf experiments/$1/figures/$2
        rm -rf experiments/$1/images/$2
        rm -rf experiments/$1/logs/$2
        rm -rf experiments/$1/models/$2
    fi
fi