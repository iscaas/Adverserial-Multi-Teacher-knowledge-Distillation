#python test.py --attack SINIFGSM
#python test.py --attack VNIFGSM

#python test.py --attack FGSM  --dataset cifar10
#python test.py --attack FFGSM --dataset cifar10
#python test.py --attack RFGSM --dataset cifar10
#python test.py --attack PGD   --dataset cifar10

#python test.py --attack FGSM  --dataset cifar100
python test.py --attack FFGSM --dataset cifar100
python test.py --attack RFGSM --dataset cifar100
python test.py --attack PGD   --dataset cifar100


#python test.py --attack SINIFGSM --dataset cifar10
#python test.py --attack VNIFGSM  --dataset cifar10

python test.py --attack SINIFGSM --dataset cifar100
python test.py --attack VNIFGSM  --dataset cifar100