#!/bin/bash
<<'COMMENT'
echo "-----------C=0-----------"
bash eval_SR.sh CPD00
bash eval_SR.sh CPD01
bash eval_SR.sh CPD02
bash eval_SR.sh CPD03
bash eval_SR.sh CPD04
bash eval_SR.sh CPD05

echo "-----------C=0.005-----------"
bash eval_SR.sh CPD10
bash eval_SR.sh CPD11
bash eval_SR.sh CPD12
bash eval_SR.sh CPD13
bash eval_SR.sh CPD14
bash eval_SR.sh CPD15

echo "-----------C=0.01-----------"
bash eval_SR.sh CPD20
bash eval_SR.sh CPD21
bash eval_SR.sh CPD22
bash eval_SR.sh CPD23
bash eval_SR.sh CPD24
bash eval_SR.sh CPD25
COMMENT

bash eval_SR.sh CPD01
bash eval_SR.sh CPD02
bash eval_SR.sh CPD03
bash eval_SR.sh CPD04
bash eval_SR.sh CPD05
bash eval_SR.sh CPD06
bash eval_SR.sh CPD07
bash eval_SR.sh CPD08
bash eval_SR.sh CPD09
bash eval_SR.sh CPD10
bash eval_SR.sh CPD11
bash eval_SR.sh CPD12
bash eval_SR.sh CPD13
bash eval_SR.sh CPD14
bash eval_SR.sh CPD15
bash eval_SR.sh CPD16
bash eval_SR.sh CPD17
bash eval_SR.sh CPD18
