#!/usr/bin/env bash 

module purge
module load applications-extra
module load cuda/8.0.44-cudNN5.1

echo 'The script is starting to run.'
# for (( i = 2; i < 7; i++ )); do
#         #statements
#         echo $i
#         sed -i "11s/.*/n_class=$i/" new_data_level_2.py
#         ipython new_data_level_2.py
#         sleep 60
# done

# echo 'Pretrain finished!' | mail -s 'Finished!' yu.li@kaust.edu.sa

# for class 1
sed -i "148s/.*/main_class=1/" level_3.py
for i in {1..18} 20 21 97; do
        #statements
        echo $i
        sed -i "149s/.*/subclass=$i/" level_3.py
        ipython level_3.py
        sleep 60
done

# for class 2
sed -i "148s/.*/main_class=2/" level_3.py
for i in {1..10}; do
        #statements
        echo $i
        sed -i "149s/.*/subclass=$i/" level_3.py
        ipython level_3.py
        sleep 60
done

# for class 3
sed -i "148s/.*/main_class=3/" level_3.py
for i in {1..8} 11; do
        #statements
        echo $i
        sed -i "149s/.*/subclass=$i/" level_3.py
        ipython level_3.py
        sleep 60
done

# for class 4
sed -i "148s/.*/main_class=4/" level_3.py
for i in {1..4} 6 99; do
        #statements
        echo $i
        sed -i "149s/.*/subclass=$i/" level_3.py
        ipython level_3.py
        sleep 60
done


# for class 5
sed -i "148s/.*/main_class=5/" level_3.py
for i in {1..5} 99; do
        #statements
        echo $i
        sed -i "149s/.*/subclass=$i/" level_3.py
        ipython level_3.py
        sleep 60
done

# for class 6
sed -i "148s/.*/main_class=6/" level_3.py
for i in {1..6}; do
        #statements
        echo $i
        sed -i "149s/.*/subclass=$i/" level_3.py
        ipython level_3.py
        sleep 60
done

echo 'All finished!' | mail -s 'Finished!' yu.li@kaust.edu.sa