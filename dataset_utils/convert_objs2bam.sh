#!/bin/bash

#################################################################################
# convert_objs2bam.sh
#
# Author: Nolan Lunscher
#
# All code is provided for research purposes only and without any warranty. 
# Any commercial use requires our consent. 
# When using the code in your research work, please cite the following paper:
#     @InProceedings{Lunscher_2017_ICCV_Workshops,
#     author = {Lunscher, Nolan and Zelek, John},
#     title = {Point Cloud Completion of Foot Shape From a Single Depth Map for Fit Matching Using Deep Learning View Synthesis},
#     booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
#     month = {Oct},
#     year = {2017}
#     }
#################################################################################

obj_folder="../../Data/raw_data/caesar-norm-wsx-fitted-meshes_foot_mesh_OBJ/"
bam_folder="../../Data/raw_data/caesar-norm-wsx-fitted-meshes_foot_mesh_BAM/"
sub_folder=""

counter=0
for filename in $obj_folder*; do
	((counter++))
	echo $counter
	# echo $filename
	name_extention=${filename##*/}
	name=${name_extention%.*}
	# echo $name_extention
	echo $name

	obj_name=$obj_folder$sub_folder$name".obj"
	egg_name=$bam_folder$sub_folder$name".egg"
	bam_name=$bam_folder$sub_folder$name".bam"

	if [ -a $obj_name ]; then # if obj file exists

		if [ ! -e $bam_name ]; then # if bam file doesnt already exist

			# needed for those that cause errors due to nan
			# replace nan with 0.00 into model.obj2.obj
			if grep -q "nan" $obj_name; then
				while read a; do echo ${a//nan/0.00} ; done < $obj_name > $obj_name"2.obj"
				obj2egg $obj_name"2.obj" -o $egg_name
				rm $obj_name"2.obj"
			else
				obj2egg $obj_name -o $egg_name
			fi

			egg2bam $egg_name -o $bam_name

			rm $egg_name
		fi
	else
		echo "Obj File not found: "$obj_name
	fi
done



