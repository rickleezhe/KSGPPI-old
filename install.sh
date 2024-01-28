#!/bin/bash

cd esmmodel
chmod 777 ./download.sh
bash ./download.sh

cd ../graph-encoding/multispecies/
java -jar FileUnion.jar ./mu_graph ./graph.emb.npz

cd ../model/
java -jar FileUnion.jar ./uniprot/multispecies ./model.pkl

cd uniprot
tar -zxvf uniprot_human_2023_07_10.tar.gz
rm -f uniprot_human_2023_07_10.tar.gz