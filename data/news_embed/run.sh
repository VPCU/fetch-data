#!/bin/bash
for i in {1..12}
do
  python emb.py -y 2020 -m $i
done

for i in {1..10}
do
  python emb.py -y 2021 -m $i
done
